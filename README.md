# nlp-notes
Overview of Natural Language Processing (NLP) neural network models

*All of the excellent figures (and some of the wording) are taken from the following references*:
- https://towardsdatascience.com/illustrated-guide-to-lstms-and-gru-s-a-step-by-step-explanation-44e9eb85bf21
- http://jalammar.github.io/illustrated-bert/
- http://colah.github.io/posts/2015-09-NN-Types-FP/
- https://www.researchgate.net/publication/327645964_Unsupervised_Suicide_Note_Classification
- https://blog.floydhub.com/attention-mechanism/
- https://towardsdatascience.com/illustrated-self-attention-2d627e33b20a

### Recurrent Neural Network (RNN)
The RNN is designed to be the simplest implementation of a neural network architecture for interpreting sequence and time-series data. As each token (word, nucleic acid, amino acid...) is read, the token (input) is concatenated with a context vector (previous hidden state), and interpreted by the neural network with a Tanh activation function:

![RNN_cell](figures/RNN_cell.gif)

The updated context vector (new hidden state) becomes the previous hidden state for interpreting the next token:

![RNN_passing](figures/RNN_passing.gif)

After interpreting the full sequence of tokens, the final context vector (final hidden state) represents the neural networks understanding of the sequence.

### Long Short-Term Memory (LSTM)
One of the major drawbacks with the RNN is it suffers from long-term memory loss (the context vector will quickly lose memory of tokens it saw a long time ago). We can remedy this by explicitly giving the neural network control over which memories to keep and which to forget. The input and previous hidden state are concatenated as before, but they are first passed through a *forget gate* with a sigmoid output between 0 (forget it) and 1 (remember it):

![LSTM_forget_gate](figures/LSTM_forget_gate.gif)

Next, the *input gate* decides what content from the hidden state to store in long-term memory with parallel sigmoid and Tanh activations (outputs between 0 to 1, and -1 to 1, respectively):

![LSTM_input_gate](figures/LSTM_input_gate.gif)

Now we can calculate our long-term memory storage *cell state* by multiplying the forget gate and adding the input gate:

![LSTM_cell_state](figures/LSTM_cell_state.gif)

Finally, the *output gate* combines the learned long-term memory with the hidden state to give our new hidden state:

![LSTM_output_gate](figures/LSTM_output_gate.gif)

### Gated Recurrent Unit (GRU)
The LSTM is a thorough, yet complex way to store long-term memory. This model can be simplified without sacrificing much in performance. The GRU removes the cell state, and only contains two gates:
- Update gate: Replaces the forget and input gates of an LSTM, deciding what information to remember and what to throw away
- Reset gate: Another gate to decide how much past information to forget

![GRU](figures/GRU.png)

### Bidirectional LSTM (BiLSTM)
Why should a machine be restricted to reading a sequence in one direction? In many cases (gene and protein sequences included) a great deal of information about a token is lost without combining information from the left and right sides:

![BiLSTM](figures/BiLSTM.png)

### Attention Mechanism
While LSTMs and GRUs are resistant to long-term memory loss, they are far from perfect and fail to retain connections between distant tokens in long sequences. Attention solves this problem by creating connections between tokens that do not decay with separation distance. The context vector is calculated as a weighted sum of all hiddens states followed by softmax activation, as opposed to simply assigning the current hidden state as the context vector:

![BiLSTM_attention](figures/BiLSTM_attention.png)

How the hidden states are weighted depends upon the specific application:
- Sequence-to-sequence model: A sequence is passed through an encoder, and the final hidden state is passed through a decoder for translation. The attention weights are a function of [1] the previous decoder hidden state and [2] each of the encoder hidden states. This allows sequence-to-sequence models to focus on the most relevant token(s) from the untranslated sequence at each step of the decoding (translation) process:
  
  ![sequence_to_sequence](figures/sequence_to_sequence.png)
  
- Self attention model: The attention model can also create distant connections between tokens belonging to the same sequence, thus eliminating the long-term memory loss problems of the LSTM/GRU models altogether. First, *keys*, *values*, and *queries* are computed from the inputs:
  
  ![self_attention_key_value_query](figures/self_attention_key_value_query.gif)
  
  Starting with the first input, its *query* is used to calculate attention scores using the *keys*, which get multiplied by the *values* :
  
  ![self_attention_calculation](figures/self_attention_calculation.gif)

  Summing the weighted results gives the output for the first input:

  ![self_attention_output](figures/self_attention_output.gif)

  And this process is continued for the remaining inputs...

### Transformers
What if we could do away with the sequential nature of the RNN/LSTM/GRU altogether and model sequence data purely with self attention? 
## Embedding
Embedding is the process of converting a token into a fixed length numerical vector. Nearly all NLP tasks can be broken down into a two-step process of [1] token embedding (TOKENIZATION?) with an unsupervised NLP model pre-trained on a database of sequences, and [2] supervised learning to complete the task at hand. The quality of the embedder is a critical component of how well the task can be learned.

### Word2Vec and GloVe
The NLP community quickly realized it was better to pre-train embeddings with unsupervised learning on large text databases, as opposed to learning them on the fly during a supervised learning task on what was frequently a small dataset. By training unsupervised models on a large corpus of text, models such as Word2Vec and GloVe learned useful word embeddings for a variety of NLP tasks.

### ELMo
Word2Vec and GloVe learn fixed numerical embeddings for words. The shortcoming of this approach is that the numerical embedding for a token should depend on the context! For example, the word "bass" should have very different embeddings for the sentences "I caught a bass" and "turn up the bass".

ELMo is an unsupervised multi-layered LSTM trained to predict the next token in a sequence:

![ELMo_LSTM](figures/ELMo_LSTM.png)

ELMo does this in both directions in order to capture context from the full sequence in its embedding, making it a bi-directional LSTM:

![ELMo_BiLM](figures/ELMo_BiLM.png)

The hidden layers from the forward and backward passes are concatenated, and are suitably aggregated for the NLP task at hand:

![ELMo_embedding](figures/ELMo_embedding.png)