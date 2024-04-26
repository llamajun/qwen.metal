/* Apple Metal inference for Llama-2 and Qwen2 Transformer model in C */

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <fcntl.h>
#include <sys/time.h>
#include <unistd.h>
#include <sys/mman.h>

#include "llm-metal.h"

#define LLAMA2 1
#define QWEN2 2

int model_type = LLAMA2;
int debug = 0;

// Model hyper-parameters
const float ROPE_THETA = 1000000.0f; // RoPE theta parameter

// ----------------------------------------------------------------------------
// Transformer model

typedef struct {
    int dim; // transformer dimension
    int hidden_dim; // for ffn layers
    int n_layers; // number of layers
    int n_heads; // number of query heads
    int n_kv_heads; // number of key/value heads (can be < query heads because of multiquery)
    int vocab_size; // vocabulary size, usually 256 (byte-level)
    int seq_len; // max sequence length
} Config;

typedef struct {
    // token embedding table
    float* token_embedding_table;    // (vocab_size, dim)
    // weights for rmsnorms
    float* rms_att_weight; // (layer, dim) rmsnorm weights
    float* rms_ffn_weight; // (layer, dim)
    // weights and biases for matmuls. note dim == n_heads * head_size
    float* wq; // (layer, dim, n_heads * head_size)
    float* bq; // (layer, n_heads * head_size), only for qwen2
    float* wk; // (layer, dim, n_kv_heads * head_size)
    float* bk; // (layer, n_kv_heads * head_size), only for qwen2
    float* wv; // (layer, dim, n_kv_heads * head_size)
    float* bv; // (layer, n_kv_heads * head_size), only for qwen2
    float* wo; // (layer, n_heads * head_size, dim)
    // weights for ffn
    float* w1; // (layer, hidden_dim, dim)
    float* w2; // (layer, dim, hidden_dim)
    float* w3; // (layer, hidden_dim, dim)
    // final rmsnorm
    float* rms_final_weight; // (dim,)
    // (optional) classifier weights for the logits, on the last layer
    float* wcls;
} TransformerWeights;

typedef struct {
    // current wave of activations
    float *x; // activation at current time stamp (dim,)
    float *xb; // same, but inside a residual branch (dim,)
    float *xb2; // an additional buffer just for convenience (dim,)
    float *hb; // buffer for hidden dimension in the ffn (hidden_dim,)
    float *hb2; // buffer for hidden dimension in the ffn (hidden_dim,)
    float *q; // query (dim,)
    float *k; // key (dim,)
    float *v; // value (dim,)
    float *att; // buffer for scores/attention values (n_heads, seq_len)
    float *logits; // output logits
    // kv cache
    float* key_cache;   // (layer, seq_len, dim)
    float* value_cache; // (layer, seq_len, dim)
} RunState;

typedef struct {
    Config config; // the hyperparameters of the architecture (the blueprint)
    TransformerWeights weights; // the weights of the model
    RunState state; // buffers for the "wave" of activations in the forward pass
    // some more state needed to properly clean up the memory mapping (sigh)
    int fd; // file descriptor for memory mapping
    float* data; // memory mapped data pointer
    ssize_t file_size; // size of the checkpoint file in bytes
} Transformer;

void malloc_run_state(RunState* s, Config* p) {
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    lm_alloc(&s->x, p->dim * sizeof(float));
    lm_alloc(&s->xb, p->dim * sizeof(float));
    lm_alloc(&s->xb2, p->dim * sizeof(float));
    lm_alloc(&s->hb, p->hidden_dim * sizeof(float));
    lm_alloc(&s->hb2, p->hidden_dim * sizeof(float));
    lm_alloc(&s->q, p->dim * sizeof(float));
    lm_alloc(&s->key_cache, p->n_layers * p->seq_len * kv_dim * sizeof(float));
    lm_alloc(&s->value_cache, p->n_layers * p->seq_len * kv_dim * sizeof(float));
    lm_alloc(&s->att, p->n_heads * p->seq_len * sizeof(float));
    lm_alloc(&s->logits, p->vocab_size * sizeof(float));
    // ensure all mallocs went fine
    if (!s->x || !s->xb || !s->xb2 || !s->hb || !s->hb2 || !s->q
     || !s->key_cache || !s->value_cache || !s->att || !s->logits) {
        fprintf(stderr, "malloc failed!\n");
        exit(EXIT_FAILURE);
    }
}

void free_run_state(RunState* s) {
    lm_free(s->x);
    lm_free(s->xb);
    lm_free(s->xb2);
    lm_free(s->hb);
    lm_free(s->hb2);
    lm_free(s->q);
    lm_free(s->att);
    lm_free(s->logits);
    lm_free(s->key_cache);
    lm_free(s->value_cache);
}

// Allocate page-aligned memory for weight matrix, register a metal buffer and copy content into it
#define WEIGHT(p, len) do { \
    if (lm_alloc(&(p), (len)*sizeof(float)) == 0) { \
        fread(p, (len)*sizeof(float), 1, file); \
    } else { \
        printf("Alloc metal buffer failure\n"); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

void read_checkpoint(char* checkpoint, Config* p, TransformerWeights* w,
                     int* fd, float** data, ssize_t* file_size) {
    FILE *file = fopen(checkpoint, "rb");
    if (!file) { fprintf(stderr, "Couldn't open file %s\n", checkpoint); exit(EXIT_FAILURE); }
    // read in the config header
    if (fread(p, sizeof(Config), 1, file) != 1) { exit(EXIT_FAILURE); }
    // limit max sequence length to 4K to save memory
    if (p->seq_len > 4096) p->seq_len = 4096; 

    // negative vocab size is hacky way of signaling unshared weights. bit yikes.
    int shared_weights = p->vocab_size > 0 ? 1 : 0;
    p->vocab_size = abs(p->vocab_size);

    // read in the transformer weights
    int head_size = p->dim / p->n_heads;
    // make sure the multiplications below are done in 64bit to fit the parameter counts of 13B+ models
    unsigned long long n_layers = p->n_layers;
    WEIGHT(w->token_embedding_table, (unsigned long long)p->vocab_size * p->dim);
    WEIGHT(w->rms_att_weight, n_layers * p->dim);
    WEIGHT(w->wq, n_layers * p->dim * (p->n_heads * head_size));
    if (model_type == QWEN2) 
        WEIGHT(w->bq, p->n_layers * p->n_heads * head_size);
    WEIGHT(w->wk, n_layers * p->dim * (p->n_kv_heads * head_size));
    if (model_type == QWEN2)
        WEIGHT(w->bk, p->n_layers * p->n_kv_heads * head_size);
    WEIGHT(w->wv, n_layers * p->dim * (p->n_kv_heads * head_size));
    if (model_type == QWEN2)
        WEIGHT(w->bv, p->n_layers * p->n_kv_heads * head_size);
    WEIGHT(w->wo, n_layers * (p->n_heads * head_size) * p->dim);
    WEIGHT(w->rms_ffn_weight, n_layers * p->dim);
    WEIGHT(w->w1, n_layers * p->hidden_dim * p->dim);
    WEIGHT(w->w2, n_layers * p->dim * p->hidden_dim);
    WEIGHT(w->w3, n_layers * p->hidden_dim * p->dim);
    WEIGHT(w->rms_final_weight, p->dim);
    fseek(file, p->seq_len * head_size / 2, SEEK_CUR); // skip what used to be freq_cis_real (for RoPE)
    fseek(file, p->seq_len * head_size / 2, SEEK_CUR); // skip what used to be freq_cis_imag (for RoPE)
    if (shared_weights) 
        w->wcls = w->token_embedding_table;
    else 
        WEIGHT(w->wcls, p->dim * p->vocab_size);

    fclose(file);
}

void build_transformer(Transformer *t, char* checkpoint_path) {
    // read in the Config and the Weights from the checkpoint
    read_checkpoint(checkpoint_path, &t->config, &t->weights, &t->fd, &t->data, &t->file_size);
    // allocate the RunState buffers
    malloc_run_state(&t->state, &t->config);
}

void free_transformer(Transformer* t) {
    // close the memory mapping
    if (t->data != MAP_FAILED) { munmap(t->data, t->file_size); }
    if (t->fd != -1) { close(t->fd); }
    // free the RunState buffers
    free_run_state(&t->state);
}

void softmax(float* x, int size) {
    // find max value (for numerical stability)
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }
    // exp and sum
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    // normalize
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

void dump(char* name, float* x, int size) {
    if (!debug) return;
    printf("%s: [", name);
    for (int i = 0; i < size; i++) {
        if (i == 3 && size > 6) {
            printf("... ");
            i = size - 3;
        }
        printf("%.4f ", x[i]);
    }
    printf("]\n");
}

float* forward(Transformer* transformer, int token, int pos) {

    // a few convenience variables
    Config* p = &transformer->config;
    TransformerWeights* w = &transformer->weights;
    RunState* s = &transformer->state;
    float *x = s->x;
    int dim = p->dim;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int kv_mul = p->n_heads / p->n_kv_heads; // integer multiplier of the kv sharing in multiquery
    int hidden_dim =  p->hidden_dim;
    int head_size = dim / p->n_heads;

    // copy the token embedding into x
    float* content_row = w->token_embedding_table + token * dim;
    memcpy(x, content_row, dim*sizeof(float));

    if (debug) printf("token %d: ", token);
    dump("", x, dim);

    // forward all the layers
    for(unsigned long long l = 0; l < p->n_layers; l++) {
        if (debug) printf("== layer %llu ==\n", l);

        // key and value point to the kv cache
        int loff = l * p->seq_len * kv_dim; // kv cache layer offset for convenience
        int koff = loff + pos * kv_dim;
        int voff = loff + pos * kv_dim;
        s->k = s->key_cache + koff;
        s->v = s->value_cache + voff;

        // attention rmsnorm
        // dump("layernorm_weight=", w->rms_att_weight + l*dim, dim);
        lm_rmsnorm(s->xb, x, w->rms_att_weight, dim, model_type == QWEN2 ? 1e-6f : 1e-5f, l*dim);

        lm_execute();
        dump("rmsnorm=", s->xb, dim);

        // qkv matmuls for this position
        lm_gemv(s->q, s->xb, w->wq, dim, dim, 0, l*dim*dim);            // matmul
        lm_gemv(s->key_cache, s->xb, w->wk, dim, kv_dim, koff, l*dim*kv_dim);     
        lm_gemv(s->value_cache, s->xb, w->wv, dim, kv_dim, voff, l*dim*kv_dim);
        if (model_type == QWEN2) {
            lm_add(s->q, s->q, w->bq, dim, 0, 0, l*dim);                    // bias
            lm_add(s->key_cache, s->key_cache, w->bk, kv_dim, koff, koff, l*kv_dim);
            lm_add(s->value_cache, s->value_cache, w->bv, kv_dim, voff, voff, l*kv_dim);
        }

        lm_execute();
        dump("q", s->q, dim);
        dump("k", s->k, kv_dim);
        dump("v", s->v, kv_dim);

        // RoPE relative positional encoding: complex-valued rotate q and k in each head

        if (model_type == QWEN2) 
            // Weight layout is different: https://github.com/juncongmoo/pyllama/issues/83
            lm_rope(1, s->q, s->key_cache, pos, p->n_heads*head_size, p->n_kv_heads*head_size, head_size, ROPE_THETA, koff);
        else 
            lm_rope(0, s->q, s->key_cache, pos, p->n_heads*head_size, p->n_kv_heads*head_size, head_size, 10000.0f, koff);

        lm_execute();
        dump("q_rope", s->q, dim);
        dump("k_rope", s->k, kv_dim);

        // multihead attention
        lm_multihead_attention(s->att, s->q, s->key_cache, head_size, p->n_heads, pos+1, loff);
        lm_softmax(s->att, s->att, pos+1, p->n_heads);
        lm_multihead_weighted_sum(s->xb, s->att, s->value_cache, head_size, p->n_heads, pos+1, loff);

        // final matmul to get the output of the attention
        lm_gemv(s->xb2, s->xb, w->wo, dim, dim, 0, l*dim*dim);

        lm_execute();
        dump("attention xb2", s->xb2, dim);
        dump("attention x0", s->x, dim);
        // printf("dim=%d\n", dim);

        // residual connection back into x
        lm_add(x, x, s->xb2, dim, 0, 0, 0);

        lm_execute();
        dump("attention x", s->x, dim);

        // ffn rmsnorm (post_attention_layernorm)
        lm_rmsnorm(s->xb, x, w->rms_ffn_weight, dim, model_type == QWEN2 ? 1e-6f : 1e-5f, l*dim);

        lm_execute();
        dump("rmsnorm2", s->xb, dim);

        // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
        // first calculate self.w1(x) and self.w3(x)
        lm_gemv(s->hb, s->xb, w->w1, dim, hidden_dim, 0, l*dim*hidden_dim);
        lm_gemv(s->hb2, s->xb, w->w3, dim, hidden_dim, 0, l*dim*hidden_dim);

        lm_execute();
        dump("hb_w1", s->hb, hidden_dim);
        dump("hb2_w3", s->hb2, hidden_dim);

        // SwiGLU non-linearity
        lm_swiglu(s->hb, s->hb, s->hb2, hidden_dim);

        lm_execute();
        dump("swiglu", s->hb, hidden_dim);

        // final matmul to get the output of the ffn
        lm_gemv(s->xb, s->hb, w->w2, hidden_dim, dim, 0, l*dim*hidden_dim);

        // residual connection
        lm_add(x, x, s->xb, dim, 0, 0, 0);
    }

    // final rmsnorm
    lm_rmsnorm(x, x, w->rms_final_weight, dim, 1e-06f, 0);

    // classifier into logits
    lm_gemv(s->logits, x, w->wcls, p->dim, p->vocab_size, 0, 0);
    lm_execute();

    dump("logits = ", s->logits, p->vocab_size);


    return s->logits;
}

// ----------------------------------------------------------------------------
// The Byte Pair Encoding (BPE) Tokenizer that translates strings <-> tokens

typedef struct {                
    int a;
    int b;
    int c;      // merge result
    int rank;   // rank = 0 for most frequent merge
} Merge;

typedef struct {
    char *str;
    int id;
} TokenIndex;

typedef struct {
    char **vocab;
    TokenIndex *sorted_vocab;
    float *vocab_scores;        // only for llama
    int max_token_length;       // only for llama
    Merge *merges;              // only for qwen
    int vocab_size;
    int merge_size;
    unsigned char byte_pieces[512]; // stores all single-byte strings
} Tokenizer;

int compare_tokens(const void *a, const void *b) {
    return strcmp(((TokenIndex*)a)->str, ((TokenIndex*)b)->str);
}

int compare_merge(const void *a, const void *b) {
    if (((Merge*)a)->a == ((Merge*)b)->a)
        return ((Merge*)a)->b - ((Merge*)b)->b;
    else
        return ((Merge*)a)->a - ((Merge*)b)->a;
}

// QWEN2 uses a merges(huggingface)-style tokenzier, while llama2.c uses sentencepiece
// style tokenizer. So we load different data depending on model_type.
// vocab_size is needed for original llama.c data format.
void build_tokenizer(Tokenizer* t, char* tokenizer_path, int vocab_size) {
    FILE *file = fopen(tokenizer_path, "rb");
    if (!file) { fprintf(stderr, "couldn't load %s\n", tokenizer_path); exit(EXIT_FAILURE); }

    // read vocab table
    if (model_type == LLAMA2) t->vocab_size = vocab_size;
    else if (fread(&t->vocab_size, sizeof(int), 1, file) != 1) {
        fprintf(stderr, "failed read vocab_size\n"); 
        exit(EXIT_FAILURE); 
    }
    t->vocab = (char**)malloc(t->vocab_size * sizeof(char*));
    if (model_type == LLAMA2) {
        t->vocab_scores = (float*)malloc(vocab_size * sizeof(float));
        if (fread(&t->max_token_length, sizeof(int), 1, file) != 1) { fprintf(stderr, "failed read\n"); exit(EXIT_FAILURE); }
        for (int i = 0; i < 256; i++) {
            t->byte_pieces[i * 2] = (unsigned char)i;
            t->byte_pieces[i * 2 + 1] = '\0';
        }
    }
    for (int i = 0; i < t->vocab_size; i++) {
        unsigned int len = 0;
        if (model_type == LLAMA2 && fread(t->vocab_scores + i, sizeof(float), 1, file) != 1)
            { fprintf(stderr, "failed read vocab score\n"); exit(EXIT_FAILURE);}
        if (fread(&len, model_type == QWEN2 ? sizeof(unsigned short) : sizeof(int), 1, file) != 1) {fprintf(stderr, "failed read vocab len %d\n", i); exit(EXIT_FAILURE); }
        t->vocab[i] = (char *)malloc(len + 1);
        if (fread(t->vocab[i], len, 1, file) != 1) {fprintf(stderr, "failed read vocab %d of len %d\n", i, len); exit(EXIT_FAILURE); }
        t->vocab[i][len] = '\0';
    }

    // sort the vocabulary for efficient lookup
    t->sorted_vocab = malloc(t->vocab_size * sizeof(TokenIndex));
    for (int i = 0; i < t->vocab_size; i++) {
        t->sorted_vocab[i].str = t->vocab[i];
        t->sorted_vocab[i].id = i;
    }
    qsort(t->sorted_vocab, t->vocab_size, sizeof(TokenIndex), compare_tokens);

    if (model_type == QWEN2) {
        // read merge table
        if (fread(&t->merge_size, sizeof(int), 1, file) != 1) { fprintf(stderr, "failed read merge_size\n"); exit(EXIT_FAILURE); }
        t->merges = (Merge *)malloc(t->merge_size * sizeof(Merge));
        int *merge_raw = (int *)malloc(t->merge_size * sizeof(int) * 3);
        if (fread(merge_raw, t->merge_size * sizeof(int) * 3, 1, file) != 1) {fprintf(stderr, "failed read merge_raw\n"); exit(EXIT_FAILURE); }
        for (int i = 0; i < t->merge_size; i++) {
            t->merges[i].a = merge_raw[i*3];
            t->merges[i].b = merge_raw[i*3+1];
            t->merges[i].c = merge_raw[i*3+2];
            t->merges[i].rank = i;
        }
        free(merge_raw);

        // sort merge table
        qsort(t->merges, t->merge_size, sizeof(Merge), compare_merge);
    }

    fclose(file);
}

void free_tokenizer(Tokenizer *t) {
    for (int i = 0; i < t->vocab_size; i++) { free(t->vocab[i]); }
    free(t->vocab);
    free(t->sorted_vocab);    
    free(t->merges);
    if (model_type == LLAMA2) free(t->vocab_scores);
}

char* decode(Tokenizer* t, int prev_token, int token) {
    char *piece = t->vocab[token];
    if (model_type == QWEN2) return piece;      // Qwen does not need more processing
    // following BOS (1) token, sentencepiece decoder strips any leading whitespace (see PR #89)
    if (prev_token == 1 && piece[0] == ' ') { piece++; }
    // careful, some tokens designate raw bytes, and look like e.g. '<0x01>'
    // parse this and convert and return the actual byte
    unsigned char byte_val;
    if (sscanf(piece, "<0x%02hhX>", &byte_val) == 1) {
        piece = (char*)t->byte_pieces + byte_val * 2;
    }
    return piece;
}

void safe_printf(char *piece) {
    // piece might be a raw byte token, and we only want to print printable chars or whitespace
    // because some of the other bytes can be various control codes, backspace, etc.
    if (piece == NULL) { return; }
    if (piece[0] == '\0') { return; }
    if (piece[1] == '\0') {
        unsigned char byte_val = piece[0];
        if (!(isprint(byte_val) || isspace(byte_val))) {
            return; // bad byte, don't print it
        }
    }
    printf("%s", piece);
}

int str_lookup(char *str, TokenIndex *sorted_vocab, int vocab_size) {
    // efficiently find the perfect match for str in vocab, return its index or -1 if not found
    TokenIndex tok = { .str = str }; // acts as the key to search for
    TokenIndex *res = bsearch(&tok, sorted_vocab, vocab_size, sizeof(TokenIndex), compare_tokens);
    return res != NULL ? res->id : -1;
}

void encode(Tokenizer *t, char *text, int *tokens, int *n_tokens) {
    if (text == NULL) {fprintf(stderr, "cannot encode NULL text\n"); exit(EXIT_FAILURE);}

    char* str_buffer;
    if (model_type == QWEN2)
        str_buffer = malloc(16);
    else 
        str_buffer = malloc((t->max_token_length*2 +1 +2) * sizeof(char));
    size_t str_len = strlen(text);

    // start at 0 tokens
    *n_tokens = 0;

    if (model_type == LLAMA2) {
        tokens[(*n_tokens)++] = 1;    // add BOS(=1) token

        // add_dummy_prefix is true by default
        // so prepend a dummy prefix token to the input string, but only if text != ""
        // TODO: pretty sure this isn't correct in the general case but I don't have the
        // energy to read more of the sentencepiece code to figure out what it's doing
        if (text[0] != '\0') {
            int dummy_prefix = str_lookup(" ", t->sorted_vocab, t->vocab_size);
            tokens[(*n_tokens)++] = dummy_prefix;
        }    
    }

    // Okay UTF-8 time. This will get messy. Here is the reference from Wikipedia:
    // Code point ↔ UTF-8 conversion
    // First code point	Last code point	Byte 1	Byte 2	Byte 3	Byte 4
    // U+0000	U+007F	    0xxxxxxx
    // U+0080	U+07FF	    110xxxxx	10xxxxxx
    // U+0800	U+FFFF	    1110xxxx	10xxxxxx	10xxxxxx
    // U+10000	U+10FFFF    11110xxx	10xxxxxx	10xxxxxx	10xxxxxx

    // process the raw (UTF-8) byte sequence of the input string
    for (char *c = text; *c != '\0'; c++) {

        // reset buffer if the current byte is ASCII or a leading byte
        // 0xC0 is 11000000, so (*c & 0xC0) keeps the first 2 bits and zeros the rest
        // 0x80 is 10000000
        // in UTF-8, all continuation bytes start with "10" in first two bits
        // so in English this is: "if this byte is not a continuation byte"
        if ((*c & 0xC0) != 0x80) {
            // this byte must be either a leading byte (11...) or an ASCII char (0x...)
            // => reset our location, as we're starting a new UTF-8 codepoint
            str_len = 0;
        }

        // check Qwen special tokens: <|im_start|>, <|im_end|>, <|endoftext|>
        if (*c == '<') {
            if (strncmp(c, "<|im_start|>", 12) == 0) {
                tokens[(*n_tokens)++] = 151644;
                str_len = 0;
                c += 12-1;
                continue;
            } else if (strncmp(c, "<|im_end|>", 10) == 0) {
                tokens[(*n_tokens)++] = 151645;
                str_len = 0;
                c += 10-1;
                continue;
            } else if (strncmp(c, "<|endoftext|>", 13) == 0) {
                tokens[(*n_tokens)++] = 151643;
                str_len = 0;
                c += 13-1;
                continue;
            }
        }

        // append the current byte to the buffer
        str_buffer[str_len++] = *c; // ++ is post-increment, incremented after this line
        str_buffer[str_len] = '\0';

        // while the next character is a continuation byte, continue appending
        // but if there are too many of them, just stop to avoid overruning str_buffer size.
        if ((*(c+1) & 0xC0) == 0x80 && str_len < 4) {
            continue;
        }

        // ok c+1 is not a continuation byte, so we've read in a full codepoint
        int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);

        if (id != -1) {
            // we found this codepoint in vocab, add it as a token
            tokens[(*n_tokens)++] = id;
        } else {
            // byte_fallback encoding: just encode each byte as a token
            // +3 is here because the first 3 vocab elements are <unk>, <s>, </s>
            // so the individual bytes only start at index 3
            for (int i=0; i < str_len; i++) {
                tokens[(*n_tokens)++] = (unsigned char)str_buffer[i] + 3;
            }
        }
        str_len = 0; // protect against a sequence of stray UTF8 continuation bytes
    }

    if (debug) {
        for (int i = 0; i < *n_tokens; i++) 
            printf("%d ", tokens[i]);
        printf("\n");
    }

    // merge the best consecutive pair each iteration, according the scores in vocab_scores
    while (1) {
        int best_rank = INT32_MAX;      // only for qwen
        float best_score = -1e10;       // only for llama
        int best_id = -1;
        int best_idx = -1;

        for (int i=0; i < (*n_tokens-1); i++) {
            if (model_type == QWEN2) {
                // check if we can merge the pair (tokens[i], tokens[i+1])
                Merge merge = {tokens[i], tokens[i+1], 0};
                Merge *res = bsearch(&merge, t->merges, t->merge_size, sizeof(Merge), compare_merge);
                if (res && res->rank < best_rank) {
                    best_rank = res->rank;
                    best_id = res->c;
                    best_idx = i;
                }
            } else {
                // check if we can merge the pair (tokens[i], tokens[i+1])
                sprintf(str_buffer, "%s%s", t->vocab[tokens[i]], t->vocab[tokens[i+1]]);
                int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
                if (id != -1 && t->vocab_scores[id] > best_score) {
                    // this merge pair exists in vocab! record its score and position
                    best_score = t->vocab_scores[id];
                    best_id = id;
                    best_idx = i;
                }                
            }
        }

        // printf("%d(%d) ", best_idx, best_id);
        if (best_idx == -1) {
            break; // we couldn't find any more pairs to merge, so we're done
        }

        // merge the consecutive pair (best_idx, best_idx+1) into new token best_id
        tokens[best_idx] = best_id;
        // delete token at position best_idx+1, shift the entire sequence back 1
        for (int i = best_idx+1; i < (*n_tokens-1); i++) {
            tokens[i] = tokens[i+1];
        }
        (*n_tokens)--; // token length decreased
    }

    // add optional EOS (=2) token, if desired
    // if (eos) tokens[(*n_tokens)++] = 2;
    
    free(str_buffer);
}

void dump_tokens(Tokenizer *t, int *tokens, int n_tokens) {
    if (!debug) return;
    printf("tokens: [");
    for (int i = 0; i < n_tokens; i++) {
        printf("%d'%s' ", tokens[i], t->vocab[tokens[i]]);
    }
    printf("]\n");
}

// ----------------------------------------------------------------------------
// The Sampler, which takes logits and returns a sampled token
// sampling can be done in a few ways: greedy argmax, sampling, top-p sampling

typedef struct {
    float prob;
    int index;
} ProbIndex; // struct used when sorting probabilities during top-p sampling

typedef struct {
    int vocab_size;
    ProbIndex* probindex; // buffer used in top-p sampling
    float temperature;
    float topp;
    unsigned long long rng_state;
} Sampler;

int sample_argmax(float* probabilities, int n) {
    // return the index that has the highest probability
    int max_i = 0;
    float max_p = probabilities[0];
    for (int i = 1; i < n; i++) {
        if (probabilities[i] > max_p) {
            max_i = i;
            max_p = probabilities[i];
        }
    }
    return max_i;
}

int sample_mult(float* probabilities, int n, float coin) {
    // sample index from probabilities (they must sum to 1!)
    // coin is a random number in [0, 1), usually from random_f32()
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += probabilities[i];
        if (coin < cdf) {
            return i;
        }
    }
    return n - 1; // in case of rounding errors
}

int compare(const void* a, const void* b) {
    ProbIndex* a_ = (ProbIndex*) a;
    ProbIndex* b_ = (ProbIndex*) b;
    if (a_->prob > b_->prob) return -1;
    if (a_->prob < b_->prob) return 1;
    return 0;
}

int sample_topp(float* probabilities, int n, float topp, ProbIndex* probindex, float coin) {
    // top-p sampling (or "nucleus sampling") samples from the smallest set of
    // tokens that exceed probability topp. This way we never sample tokens that
    // have very low probabilities and are less likely to go "off the rails".
    // coin is a random number in [0, 1), usually from random_f32()

    int n0 = 0;
    // quicksort indices in descending order of probabilities
    // values smaller than (1 - topp) / (n - 1) cannot be part of the result
    // so for efficiency we crop these out as candidates before sorting
    const float cutoff = (1.0f - topp) / (n - 1);
    for (int i = 0; i < n; i++) {
        if (probabilities[i] >= cutoff) {
            probindex[n0].index = i;
            probindex[n0].prob = probabilities[i];
            n0++;
        }
    }
    qsort(probindex, n0, sizeof(ProbIndex), compare);

    // truncate the list where cumulative probability exceeds topp
    float cumulative_prob = 0.0f;
    int last_idx = n0 - 1; // in case of rounding errors consider all elements
    for (int i = 0; i < n0; i++) {
        cumulative_prob += probindex[i].prob;
        if (cumulative_prob > topp) {
            last_idx = i;
            break; // we've exceeded topp by including last_idx
        }
    }

    // sample from the truncated list
    float r = coin * cumulative_prob;
    float cdf = 0.0f;
    for (int i = 0; i <= last_idx; i++) {
        cdf += probindex[i].prob;
        if (r < cdf) {
            return probindex[i].index;
        }
    }
    return probindex[last_idx].index; // in case of rounding errors
}

void build_sampler(Sampler* sampler, int vocab_size, float temperature, float topp, unsigned long long rng_seed) {
    sampler->vocab_size = vocab_size;
    sampler->temperature = temperature;
    sampler->topp = topp;
    sampler->rng_state = rng_seed;
    // buffer only used with nucleus sampling; may not need but it's ~small
    sampler->probindex = malloc(sampler->vocab_size * sizeof(ProbIndex));
}

void free_sampler(Sampler* sampler) {
    free(sampler->probindex);
}

unsigned int random_u32(unsigned long long *state) {
    // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    return (*state * 0x2545F4914F6CDD1Dull) >> 32;
}
float random_f32(unsigned long long *state) { // random float32 in [0,1)
    return (random_u32(state) >> 8) / 16777216.0f;
}

int sample(Sampler* sampler, float* logits) {
    // sample the token given the logits and some hyperparameters
    int next;
    if (sampler->temperature == 0.0f) {
        // greedy argmax sampling: take the token with the highest probability
        next = sample_argmax(logits, sampler->vocab_size);
    } else {
        // apply the temperature to the logits
        for (int q=0; q<sampler->vocab_size; q++) { logits[q] /= sampler->temperature; }
        // apply softmax to the logits to get the probabilities for next token
        softmax(logits, sampler->vocab_size);
        // flip a (float) coin (this is our source of entropy for sampling)
        float coin = random_f32(&sampler->rng_state);
        // we sample from this distribution to get the next token
        if (sampler->topp <= 0 || sampler->topp >= 1) {
            // simply sample from the predicted probability distribution
            next = sample_mult(logits, sampler->vocab_size, coin);
        } else {
            // top-p (nucleus) sampling, clamping the least likely tokens to zero
            next = sample_topp(logits, sampler->vocab_size, sampler->topp, sampler->probindex, coin);
        }
    }
    return next;
}

// ----------------------------------------------------------------------------
// utilities: time

long time_in_ms() {
    // return time in milliseconds, for benchmarking the model speed
    struct timespec time;
    clock_gettime(CLOCK_REALTIME, &time);
    return time.tv_sec * 1000 + time.tv_nsec / 1000000;
}

// ----------------------------------------------------------------------------
// generation loop

void generate(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler, char *prompt, int steps) {
    char *empty_prompt = "";
    if (prompt == NULL) { prompt = empty_prompt; }

    // encode the (string) prompt into tokens sequence
    int num_prompt_tokens = 0;
    int* prompt_tokens = (int*)malloc((strlen(prompt)+3) * sizeof(int)); // +3 for '\0', ?BOS, ?EOS
    // encode(tokenizer, prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
    encode(tokenizer, prompt, prompt_tokens, &num_prompt_tokens);
    if (num_prompt_tokens < 1) {
        fprintf(stderr, "something is wrong, expected at least 1 prompt token\n");
        exit(EXIT_FAILURE);
    }

    // start the main loop
    long start = 0;  // used to time our code, only initialized after first iteration
    int next;        // will store the next token in the sequence
    int token = prompt_tokens[0]; // kick off with the first token in the prompt
    int pos = 0;     // position in the sequence
    while (pos < steps) {

        // forward the transformer to get logits for the next token
        float* logits = forward(transformer, token, pos);

        // advance the state machine
        if (pos < num_prompt_tokens - 1) {
            // if we are still processing the input prompt, force the next prompt token
            next = prompt_tokens[pos + 1];
        } else {
            // otherwise sample the next token from the logits
            next = sample(sampler, logits);
        }
        pos++;

        // print the token as string, decode it with the Tokenizer object
        // char* piece = decode(tokenizer, token, next);
        char* piece = model_type == QWEN2 ? decode(tokenizer, 0, token) : decode(tokenizer, token, next);
        safe_printf(piece);     // same as printf("%s", piece), but skips "unsafe" bytes
        fflush(stdout);
        token = next;

        // data-dependent terminating condition: the BOS (=1) token delimits sequences
        if (model_type == QWEN2) {
            if (next == 151645) 
                break;
        } else if (next == 1)
            break;

        // init the timer here because the first iteration can be slower
        if (start == 0) { start = time_in_ms(); }
    }
    printf("\n");

    // report achieved tok/s (pos-1 because the timer starts after first iteration)
    if (pos > 1) {
        long end = time_in_ms();
        fprintf(stderr, "achieved tok/s: %f\n", (pos-1) / (double)(end-start)*1000);
    }

    free(prompt_tokens);
}

void read_stdin(const char* guide, char* buffer, size_t bufsize) {
    // read a line from stdin, up to but not including \n
    printf("%s", guide);
    if (fgets(buffer, bufsize, stdin) != NULL) {
        size_t len = strlen(buffer);
        if (len > 0 && buffer[len - 1] == '\n') {
            buffer[len - 1] = '\0'; // strip newline
        }
    }
}

// ----------------------------------------------------------------------------
// chat loop
// I manually inspected the tokens for a few chat conversations compared to
// python reference and that seemed ok, but this was not thoroughly tested and
// is not safely implemented, it's more a proof of concept atm.

void chat(Transformer *transformer, Tokenizer *tokenizer, Sampler *sampler,
          char *cli_user_prompt, char *cli_system_prompt, int steps) {

    // buffers for reading the system prompt and user prompt from stdin
    // you'll notice they are soomewhat haphazardly and unsafely set atm
    char system_prompt[512];
    char user_prompt[512];
    char rendered_prompt[1152];
    int num_prompt_tokens = 0;
    int* prompt_tokens = (int*)malloc(1152 * sizeof(int));
    int user_idx;

    // start the main loop
    int8_t user_turn = 1; // user starts
    int next;        // will store the next token in the sequence
    int token;       // stores the current token to feed into the transformer
    int prev_token;
    int pos = 0;     // position in the sequence
    int start_pos = 0;
    int EOS = model_type == QWEN2 ? 151645 : 2; // EOS token
    struct timeval stop, start;
    gettimeofday(&start, NULL);    
    while (pos < steps) {

        // when it is the user's turn to contribute tokens to the dialog...
        if (user_turn) {
            // get the (optional) system prompt at position 0
            if (pos == 0) {
                // at position 0, the user can also contribute a system prompt
                if (cli_system_prompt == NULL) {
                    // system prompt was not passed in, attempt to get it from stdin
                    read_stdin("Enter system prompt (optional): ", system_prompt, sizeof(system_prompt));
                } else {
                    // system prompt was passed in, use it
                    strcpy(system_prompt, cli_system_prompt);
                }
            }
            // get the user prompt
            if (pos == 0 && cli_user_prompt != NULL) {
                // user prompt for position 0 was passed in, use it
                strcpy(user_prompt, cli_user_prompt);
            } else {
                // otherwise get user prompt from stdin
                read_stdin("User: ", user_prompt, sizeof(user_prompt));
                start_pos = pos;
                gettimeofday(&start, NULL);
            }
            // render user/system prompts into the Llama 2 Chat schema
            if (pos == 0 && system_prompt[0] != '\0') {
                char *system_template = model_type == QWEN2 ? 
                    "<|im_start|>system\n%s<|im_end|>\n<|im_start|>user\n%s<|im_end|><|im_start|>assistant\n" :
                    "[INST] <<SYS>>\n%s\n<</SYS>>\n\n%s [/INST]";
                sprintf(rendered_prompt, system_template, system_prompt, user_prompt);
            } else {
                char *user_template = model_type == QWEN2 ?
                    "<|im_start|>user\n%s<|im_end|><|im_start|>assistant\n" :
                    "[INST] %s [/INST]";
                sprintf(rendered_prompt, user_template, user_prompt);
            }
            // encode the rendered prompt into tokens
            // encode(tokenizer, rendered_prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
            encode(tokenizer, rendered_prompt, prompt_tokens, &num_prompt_tokens);
            dump_tokens(tokenizer, prompt_tokens, num_prompt_tokens);
            user_idx = 0; // reset the user index
            user_turn = 0;
            printf("Assistant: ");
        }

        // determine the token to pass into the transformer next
        if (user_idx < num_prompt_tokens) {
            // if we are still processing the input prompt, force the next prompt token
            token = prompt_tokens[user_idx++];
        } else {
            // otherwise use the next token sampled from previous turn
            token = next;
        }

        if (user_idx == num_prompt_tokens) {
            int _debug = debug;
            debug = 1;
            // dump("Final state after prefill: ", transformer->state.x, transformer->config.dim);
            debug = _debug;
        }

        // EOS (=2) token ends the Assistant turn
        if (user_idx >= num_prompt_tokens && token == EOS) {
            user_turn = 1; 
            gettimeofday(&stop, NULL);
            int us = (stop.tv_sec - start.tv_sec) * 1000000 + stop.tv_usec - start.tv_usec;
            printf("\ntook %d us. %.1f token/s\n", us, (float)(pos-start_pos) / (float)us * 1000000);
        }

        // forward the transformer to get logits for the next token
        float* logits = forward(transformer, token, pos);
        next = sample(sampler, logits);
        pos++;

        if (user_idx >= num_prompt_tokens && next != EOS) {
            // the Assistant is responding, so print its output
            char* piece = decode(tokenizer, token, next);
            safe_printf(piece); // same as printf("%s", piece), but skips "unsafe" bytes
            fflush(stdout);
        }
        // if (next == EOS) { printf("\n"); }
    }
    printf("\n");
    free(prompt_tokens);
}

// ----------------------------------------------------------------------------
// parse config.ini
// right now the only option is to set the model type:
// model=llama2 or qwen2

void trim(char *str) {
    char *start = str;
    char *end = start + strlen(str) - 1;
    // Trim leading space
    while(*start && isspace((unsigned char)*start)) 
        start++;
    // Trim trailing space
    while(end > start && isspace((unsigned char)*end)) 
        end--;
    // Write null terminator
    *(end + 1) = '\0';
    // Shift from "start" to the beginning of the string
    memmove(str, start, end - start + 2);
}

void load_config() {
    FILE *f = fopen("config.ini", "r");
    if (f == NULL) {
        fprintf(stderr, "couldn't open config.ini\n");
        exit(EXIT_FAILURE);
    }
    char *line = (char *) malloc(100);
    size_t cap = 100;
    int len;
    while ((len = getline(&line, &cap, f)) != -1) {
        char *key = line;
        char *val = NULL;
        for (int i = 0; i < len; i++) {
            if (line[i] == '=') {
                line[i] = '\0';
                val = line + i + 1;
                break;
            }
        }
        if (val == NULL) continue;
        trim(key);
        trim(val);
        if (strcmp(key, "model") == 0) {
            if (strcmp(val, "llama2") == 0)
                model_type = LLAMA2;
            else if (strcmp(val, "qwen2") == 0)
                model_type = QWEN2;
            else {
                printf("Unknown model type %s\n", val);
                exit(EXIT_FAILURE);
            }

        }       // ignore all other lines
    }
    free(line);
}

// ----------------------------------------------------------------------------
// CLI, include only if not testing
#ifndef TESTING

void error_usage() {
    fprintf(stderr, "Usage:   run <model_dir> [options]\n");
    fprintf(stderr, "Example: run qwen1.5-0.5b-chat -n 256 -i \"Once upon a time\"\n");
    fprintf(stderr, "Options:\n");
    fprintf(stderr, "  -t <float>  temperature in [0,inf], default 1.0\n");
    fprintf(stderr, "  -p <float>  p value in top-p (nucleus) sampling in [0,1] default 0.9\n");
    fprintf(stderr, "  -s <int>    random seed, default time(NULL)\n");
    fprintf(stderr, "  -n <int>    number of steps to run for, default 256. 0 = max_seq_len\n");
    fprintf(stderr, "  -i <string> input prompt\n");
    fprintf(stderr, "  -z <string> optional path to custom tokenizer\n");
    fprintf(stderr, "  -m <string> mode: generate|chat, default: generate\n");
    fprintf(stderr, "  -y <string> (optional) system prompt in chat mode\n");
    exit(EXIT_FAILURE);
}

int main(int argc, char *argv[]) {
    // initialize metal
    if (lm_init() != 0) {
        printf("metal initialization failed\n");
        return 1;
    }
    // default parameters
    char *model_path = NULL;  // e.g. model
    char *tokenizer_path = "tokenizer.bin";
    float temperature = 1.0f;   // 0.0 = greedy deterministic. 1.0 = original. don't set higher
    float topp = 0.9f;          // top-p in nucleus sampling. 1.0 = off. 0.9 works well, but slower
    int steps = 256;            // number of steps to run for
    char *prompt = NULL;        // prompt string
    unsigned long long rng_seed = 0; // seed rng with time by default
    char *mode = "generate";    // generate|chat
    char *system_prompt = NULL; // the (optional) system prompt to use in chat mode

    // poor man's C argparse so we can override the defaults above from the command line
    if (argc >= 2) { model_path = argv[1]; } else { error_usage(); }
    for (int i = 2; i < argc; i+=2) {
        // do some basic validation
        if (i + 1 >= argc) { error_usage(); } // must have arg after flag
        if (argv[i][0] != '-') { error_usage(); } // must start with dash
        if (strlen(argv[i]) != 2) { error_usage(); } // must be -x (one dash, one letter)
        // read in the args
        if (argv[i][1] == 't') { temperature = atof(argv[i + 1]); }
        else if (argv[i][1] == 'p') { topp = atof(argv[i + 1]); }
        else if (argv[i][1] == 's') { rng_seed = atoi(argv[i + 1]); }
        else if (argv[i][1] == 'n') { steps = atoi(argv[i + 1]); }
        else if (argv[i][1] == 'i') { prompt = argv[i + 1]; }
        else if (argv[i][1] == 'z') { tokenizer_path = argv[i + 1]; }
        else if (argv[i][1] == 'm') { mode = argv[i + 1]; }
        else if (argv[i][1] == 'y') { system_prompt = argv[i + 1]; }
        else { error_usage(); }
    }

    // parameter validation/overrides
    if (rng_seed <= 0) rng_seed = (unsigned int)time(NULL);
    if (temperature < 0.0) temperature = 0.0;
    if (topp < 0.0 || 1.0 < topp) topp = 0.9;
    if (steps < 0) steps = 0;

    // load configuration
    if (chdir(model_path) == -1) { fprintf(stderr, "couldn't chdir to %s\n", model_path); exit(EXIT_FAILURE); }
    load_config();          // set model_type

    // build the Transformer via the model .bin file
    Transformer transformer;
    build_transformer(&transformer, "model.bin");
    if (steps == 0 || steps > transformer.config.seq_len) steps = transformer.config.seq_len; // override to ~max length

    // build the Tokenizer via the tokenizer .bin file
    Tokenizer tokenizer;
    build_tokenizer(&tokenizer, tokenizer_path, model_type == QWEN2 ? 0 : transformer.config.vocab_size);

    // build the Sampler
    Sampler sampler;
    build_sampler(&sampler, transformer.config.vocab_size, temperature, topp, rng_seed);

    // run!
    if (strcmp(mode, "generate") == 0) {
        generate(&transformer, &tokenizer, &sampler, prompt, steps);
    } else if (strcmp(mode, "chat") == 0) {
        if (model_type == QWEN2 && system_prompt == NULL) 
            system_prompt = "You are a helpful assistant.";
        chat(&transformer, &tokenizer, &sampler, prompt, system_prompt, steps);
    } else {
        fprintf(stderr, "unknown mode: %s\n", mode);
        error_usage();
    }

    // memory and file handles cleanup
    free_sampler(&sampler);
    free_tokenizer(&tokenizer);
    free_transformer(&transformer);
    return 0;
}
#endif
