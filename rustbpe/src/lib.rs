use std::cmp::Ordering;
use std::collections::HashMap as StdHashMap;

use dary_heap::OctonaryHeap;
use fancy_regex::Regex;
use pyo3::prelude::*;

use ahash::{AHashMap, AHashSet};
use compact_str::CompactString;
use rayon::prelude::*;

// Default GPT-4 style regex pattern for splitting text
const GPT4_PATTERN: &str = r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+";

type Pair = (u32, u32);



#[derive(Clone, Debug)]

// word struct
struct Word {
    ids: Vec<u32>,
}

// word implementation
impl Word {
    #[inline] // compiler optimization
    fn new(ids: Vec<u32>) -> Self {
        Self { ids } // return new word with ids
    }

    #[inline]
    /*
        lifetime pair 'a lasts for the lifetime of the word.
        We will return an iterator that yields pairs of ids.
        Iterator is only valid for the lifetime of the word
    */
    fn pairs<'a>(&'a self) -> impl Iterator<Item = Pair> + 'a {
        // map windows of 2 ids as pairs
        self.ids.windows(2).map(|w| (w[0], w[1]))
    }
    /**
     * Find pair (a,b) in self.ids vector of token ids
     * Remove (a,b) in self.ids and update with (x, new_id), (new_id, y)
     * where x is the token before a and y is the token after b 
     * So what used to be: [x,a,b,y] becomes [x,new_id,y]
     */
    fn merge_pair(&mut self, pair: Pair, new_id: u32) -> Vec<(Pair, i32)> {
        let (a, b) = pair;
        let n = self.ids.len();
        // if n < 2 then return empty vector
        if (n < 2) {
            return Vec::new();
        }

        // the out vector is initalized with capacity u32
        let mut out: Vec<u32> = Vec::with_capacity(n);
        let mut deltas: Vec<(Pair, i32)> = Vec::with_capacity(6);

        let mut i = 0;
        while i < n {
            // if next iter will happen and ids pair is (a,b)
            if i + 1 < n && self.ids[i] == a && self.ids[i + 1] == b {
                let left = out.last().copied();
                // determine if right exists
                let mut right = None;
                if i + 2 < n {
                    right = Some(self.ids[i + 2])
                }
                if let Some(x) = left { // if left exists
                    // replace (x, a) with (x, new_id)
                    deltas.push(((x, a), -1));
                    deltas.push(((x, new_id), 1));
                }
                deltas.push(((a, b), -1));
                if let Some(y) = right { // if right exists 
                    // replace (b, y) with (new_id, y)
                    deltas.push(((b, y), -1));
                    deltas.push(((new_id, y), 1));
                }
                out.push(new_id);
                i += 2; // incr by 2
            } else {
                
                out.push(self.ids[i]);
                i += 1;
            }
        }
        self.ids = out; // update self.ids to out
        deltas // return deltas 
    }
}

#[derive(Debug,Eq)]
struct MergeJob {
    pair: Pair,
    count: u64,
    pos: AHashSet<usize>
}

impl PartialEq for MergeJob {
    // check if count and pair are equivalent in input and self 
    fn eq(&self, other: &Self) -> bool {
        self.count == other.count && self.pair == other.pair
    }
}

impl PartialOrd for MergeJob {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for MergeJob {
    // Sort items in ascending order 
    fn cmp(&self, other: &Self) -> Ordering {
        // Compare self to other, return Greater, Equal, or Less
        if self.count != other.count {
            self.count.cmp(&other.count)
        } else { // tie break with 
            other.pair.cmp(&self.pair)
        }
    }
}
/**
 * returns 2 hash maps, 
 * A) maps pairs (a,b) to number of counts[index]
 * B) maps pairs (a,b) to index where they appear
 */
#[inline]
fn count_pairs_parallel(
    words: &[Word],
    counts: &[i32],
) -> (AHashMap<Pair, i32>, AHashMap<Pair, AHashSet<usize>>) {
    // note: I suspect there is a way to optimize this with segmented scan data parallel thinking
    // https://gfxcourses.stanford.edu/cs149/fall25/lecture/dataparallel/slide_30
    words  // given vector of words, 
        .par_iter() // iterate through in parallel 
        .enumerate() // get <index, word>
        .map(|(i, w)| { // returns 2 hashmaps of count and unique index
             // `i` is index, `w` is word
            // hash map of pairs to ints, how many times pair (a,b) exists in word scaled by counts[i]
            let mut local_pc: AHashMap<Pair, i32> = AHashMap::new();
            // hash map of pairs to hash sets of ints, indexes where pair (a,b) appears
            let mut local_wtu: AHashMap<Pair, AHashSet<usize>> = AHashMap::new();
            if w.ids.len() >= 2 && counts[i] != 0 { 
                for (a, b) in w.pairs() { // pair w and loop through pairs (a,b)
                    *local_pc.entry((a,b)).or_default() += counts[i]; // increment (a,b)'s int by count[i]
                    local_wtu.entry((a,b)).or_default().insert(i); // insert i into (a,b) hash set 
                }
            }
            (local_pc,local_wtu) // return hash maps of counts of ints and local_wtu 
        })
        .reduce( // consolidate array of hashmaps into 2 global hashmaps
            || (AHashMap::new(), AHashMap::new()), 
            | (mut acc_pc, mut acc_wtu), (pc, wtu) | {
                // loop over the hash map 
                for (k,v) in pc {
                    // add local hash map key-value to global
                    *acc_pc.entry(k).or_default() += v;
                }
                // loop over the hash map
                for (k,s) in wtu {
                    // extend set of local hash map key-value to global
                    acc_wtu.entry(k).or_default().extend(s);
                }
                (acc_pc, acc_wtu)
            },
        )
}


// tokenizer
#[pyclass]
pub struct Tokenizer {
    pub merges: StdHashMap<Pair, u32>,
    pub pattern: String,
    compiled_pattern: Regex,
}

impl Tokenizer {
    fn train_core_incremental(&mut self, mut words: Vec<Word>, counts: Vec<i32>, vocab_size: u32) {
        assert!(vocab_size >= 256, "vocab size must be at least 256");
        let num_merges = vocab_size - 256;
        log::info!("Starting BPE training: {} merges to compute", num_merges);
        self.merges.clear();

        // ---- Initial pair_counts and where_to_update (parallel) ----
        log::info!("Computing init pair counts from {} unique sequences", words.len());
        let (mut pair_counts, mut where_to_update) = count_pairs_parallel(&words, &counts);
        
        // ---- Build heap ----
        log::info!("Building heap with {} unique pairs", pair_counts.len());
        let mut heap: OctonaryHeap<MergeJob> = OctonaryHeap::with_capacity(pair_counts.len());
        // .drain() converts map to iteration of k-v pairs
        for (pair, pos) in where_to_update.drain() { // pass over all positions of pair
            // get count from pair_counts 
            let count_pair = pair_counts.get(&pair); 
            // either count_pair or 0 if count_pair is None and dereference it 
            let c = *count_pair.unwrap_or(&0);
            if c > 0 { // if pair index exists, add to heap 
                heap.push(MergeJob { 
                    pair, 
                    count: c as u64, 
                    pos })
            }
        }

        // ---- Merge loop ----
        log::info!("Starting merge loop");
        let mut merges_done = 0u32;
        let mut last_log_percent = 0u32;

        while merges_done < num_merges {
            // get the top pair 
            let Some(mut top) = heap.pop() else {break; };

            // lazy refresh
            let current = *pair_counts.get(&top.pair).unwrap_or(&0);
            if top.count != current as u64 { // update top.count if it is incorrect
                top.count = current as u64; 
                if top.count > 0 { // if there are still counts, push back to the heap
                    heap.push(top);
                }
                continue;
            }
            if top.count == 0 {
                break; 
            }

            // Record merge
            let new_id = 256 + merges_done;
            // init top pair to merge array
            self.merges.insert(top.pair, new_id);
            // create updates hashmap
            let mut local_pos_updates : AHashMap<Pair, AHashSet<usize>> = AHashMap::new();
            for &word_idx in &top.pos { // pass over 
                // collect pair-count deltas and apply merge 
                let changes = words[word_idx].merge_pair(top.pair, new_id);
                for (pair,delta) in changes {
                    let delta_total = delta * counts[word_idx];
                    if delta_total != 0 {
                        *pair_counts.entry(pair).or_default() += delta_total;
                        if delta > 0 {
                            local_pos_updates.entry(pair).or_default().insert(word_idx);
                        }
                    }
                }
            }

            // add updated pair 
            for(pair, pos) in local_pos_updates {
                let cnt = *pair_counts.get(&pair).unwrap_or(&0);
                if cnt > 0 {
                    heap.push(MergeJob { 
                        pair, 
                        count: cnt as u64, 
                        pos })
                }
            }
            
            merges_done += 1;

        }
    }
}

/// The rustbpe Python module
#[pymodule]
fn rustbpe(m: &Bound<'_, PyModule>) -> PyResult<()> {
    Ok(())
}
