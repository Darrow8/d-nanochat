
## Tokenizer 

[Let's build the GPT Tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE&t=4680s), [code](https://colab.research.google.com/drive/1y0KnCFZvGVf_odSfcNAws6kcDD7HsI0L?usp=sharing#scrollTo=pkAPaUCXOhvW)

The tokenizer uses a vocabulary to convert text into a set of IDs that are chunked such that each chunk contains a useful distinct idea.


### Encoding

First we take a body of text and separate it using regex into words. UTF-8 encoding converts all characters in our words to bytes and then numbers. This is the first encoding that the tokenizer does to convert the characters into IDs. We assemble all distinct IDs into a vocabulary. Then we use byte pair encoding (BPE) to grow the vocabulary and shrink the number of characters in text until we have built based on replacing the N most common pairs of IDs with a singular new ID (N times until we reach vocab size of ~50k IDs). We merge common pairs of IDs in our text to create a new result of a sequence of IDs. We must keep a hashmap recording our new IDs to our updated pairs of IDs. 

After BPE, These IDs have been paired together so many times that they encapsulate meaningful units and we call them tokens. They are on average 4 characters. 

Special tokens are added to our vocabulary to denote certain important things like start/end of a message. 

### Decoding

To decode our tokens to text we apply our hashmap which stores all mappings of tokens to byte sequences of IDs. These byte sequences are recursively decoded until we get back to fully UTF-8 IDs. 

### Tokenizer Regex to ensure proper chunking:

This GPT4 Pattern ensures that pairs of characters and pairs in certain places within a sentence will never be merged.

`const GPT4_PATTERN: &str = r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+";`

`\p{L}` = any Unicode letter (A–Z, accented letters, Greek, Cyrillic, etc.)

`\p{N}` = any Unicode number (0–9 and other numeral systems)

`\r\n` are Windows newlines; `[\r\n]` means either carriage return or newline.

`++` and `?+` are possessive quantifiers (common in engines like PCRE/Oniguruma/ICU/regex module): they do not backtrack, which makes tokenization faster and more deterministic.

- First part: `(?i:[sdmt]|ll|ve|re)` matches 's, 'll, etc in english words
- Second part: `[^\r\n\p{L}\p{N}]?+\p{L}+` excludes newline 
- Third part: `\p{N}{1,3}` matches 1 to 3 numbers
- Fourth part: `?[^\s\p{L}\p{N}]++[\r\n]*` matches puntuation symbols like "...", "?!"
- Fifth part: `\s*[\r\n]` matches newlines
- Sixth part: `\s+(?!\S)` matches trailing whitespace
- Seventh part: matches other whitespace


## AdamW Optimizer Algorithm

[AdamW Paper](https://arxiv.org/abs/1711.05101), [Adam Paper](https://arxiv.org/abs/1412.6980)

AdamW is a form of stochastic gradient descent and improves upon SGD.

### Stochastic gradient descent (SGD)

Stochastic gradient descent (SGD) optimizes for the minimum of the loss function by iteratively updating in the direction of the negative gradient. Given our dataset and a starting point, we calculate the loss relative to a chosen datapoint and then compute the gradient of the loss for that datapoint. 

$\theta \leftarrow \theta - \alpha \nabla_\theta \ell(\theta)$


### AdamW Improvements

AdamW is the result of several key improvements on SGD: 

- We add a momentum term which keeps a running track of past gradients, similar to momentum in physics. 

$$
\begin{align*}
\beta_1 &= 0.9\\
g_t &= \nabla_\theta \ell(\theta) \\
m_t &= \beta_1 m_{t-1} + (1 - \beta_1)\, g_t
\end{align*}

$$

- We add a squared gradient term with a running sum of past squared gradients to limit the variance of the gradient update.

$$
\begin{align*}
\beta_1 &= 0.999\\

v_t = \beta_2 v_{t-1} + (1 - \beta_2)\, g_t^2 \\

\end{align*}
$$

- We add bias correction terms in order to limit the influence of momentum and squared gradient values in our first iterations.

$$
\begin{align*}
\hat{m}_t &= \frac{m_t}{1 - \beta_1^t},
\qquad
\hat{v}_t = \frac{v_t}{1 - \beta_2^t} \\
\end{align*}

$$

Below is the full AdamW algorithm. 

$$
\begin{align*}
\text{First moment (momentum):}\\
m_t &= \beta_1 m_{t-1} + (1 - \beta_1)\, g_t \\

\text{Second moment (uncentered variance):}\\
v_t &= \beta_2 v_{t-1} + (1 - \beta_2)\, g_t^2 \\

\text{Bias correction:}\\
\hat{m}_t &= \frac{m_t}{1 - \beta_1^t},
\qquad
\hat{v}_t = \frac{v_t}{1 - \beta_2^t} \\

\text{Parameter update:}\\
\theta_{t+1}
&=
\theta_t
-
\eta \, \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \varepsilon}
\end{align*}
$$


## Muon Optimizer Algorithm 

[Muon Optimizer Video](https://www.youtube.com/watch?v=bO5nvE289ec), [Keller Jordan on Muon](https://kellerjordan.github.io/posts/muon/)

Similarly to AdamW, Muon seeks to fit a model to a dataset. It is faster than AdamW in certain cases, and nanoGPT uses both AdamW and Muon in training.

Muon operates similarly to SGD+momentum except it takes the updates of SGD momentum and uses a Newton-Schulz iteration on them before applying the update. As a result Muon is faster than SGD and AdamW. 

Muon takes advantage of singular value decomposition (SVD). SVD is a principle of matrices showing that any matrix can be reduced to the product 3 matrices where matrices rotate -> stretch -> rotate.

Newton-Schulz matrix iteration finds the nearest semi-orthogonal matrix to the given matrix 