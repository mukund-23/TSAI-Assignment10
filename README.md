# Assignment 10: Looking inside nanoGPT

Take a small model and a real training loop, and make it tell you the truth about itself.

This repo contains one notebook, `assignment-10.ipynb`, which runs a shrunk nanoGPT on tiny-shakespeare and interrogates it at six points: every tensor shape in a step, a gradient verified by hand against finite differences, gradient accumulation broken on purpose, grad norm logged against loss to find a step where one moved before the other, a measured MFU with a sweep to explain it, and the number 0.1 written out bit by bit in fp32, bf16 and fp8 E4M3.

Everything below is taken from the notebook's actual output.

## Setup

| item | value |
|---|---|
| model | decoder-only transformer (nanoGPT), 4 layers, 4 heads, n_embd 128 |
| context | 128 tokens |
| tokenizer | character level, vocab 65 |
| dataset | tiny-shakespeare, 1,115,394 tokens, 90/10 train/val split |
| parameters | 813,440 total, 788,736 non-embedding |
| optimizer | AdamW, lr 3e-4 |
| hardware | Kaggle, single Tesla T4 |
| precision | fp32 throughout (float64 for the gradient check only) |
| seed | 42 |

One architecture is used for the whole notebook.

## 1. Every tensor shape in one step

A `trace()` helper prints each tensor's shape along with a line saying what each dimension means. It is off during training and switched on for exactly one forward and backward pass.

With B=32 and T=128:

| tensor | shape | what the dimensions mean |
|---|---|---|
| `input_ids` | (32, 128) | B, T. Token ids indexing into the vocab |
| `residual_in` | (32, 128, 128) | B, T, C. Token embedding plus position embedding |
| `attn.qkv` | (32, 128, 384) | B, T, 3C. Query, key and value packed on the last dim |
| `attn.q` | (32, 4, 128, 32) | B, n_head, T, head_dim. One query vector per head per position |
| `attn.scores` | (32, 4, 128, 128) | B, n_head, T_query, T_key. Affinity of every position to every earlier one |
| `attn.wv` | (32, 4, 128, 32) | B, n_head, T, head_dim. Value vectors weighted by attention |
| `attn.merged` | (32, 128, 128) | B, T, C. Heads concatenated back into one embedding |
| `attn.out` | (32, 128, 128) | B, T, C. Projected back into the residual stream |
| `mlp.hidden` | (32, 128, 512) | B, T, 4C. Widened hidden layer |
| `mlp.out` | (32, 128, 128) | B, T, C. Projected back to model dim |
| `logits` | (32, 128, 65) | B, T, vocab_size. Next-token scores at every position |
| `loss` | scalar | Mean cross entropy over B*T predictions |

The notebook also prints every parameter tensor next to the shape of its `.grad` after `backward()`, confirming that each gradient matches its parameter exactly. head_dim is 32 because n_embd 128 divided by 4 heads gives 32.

## 2. Gradient verified by hand

One scalar was picked: element (0,0) of `blocks.0.mlp.fc.weight`, the weight connecting input channel 0 to hidden unit 0 of the first block's MLP. It is one number out of 813,440.

Two independent methods were used to find the slope of the loss with respect to that weight. `backward()` computes it analytically by the chain rule without ever changing the weight. Central differences computes it by brute force: actually set the weight to w+eps, run the model forward, record the loss, then set it to w-eps, run forward again, and take the rise over the run.

The model was cast to float64 and put in `eval()` so dropout masks could not differ between the two forward passes, and the same fixed batch (B=8, T=32) was used throughout.

```
backward()  0.003214172682

     eps        finite diff    rel error
   1e-03     0.003214174839     6.71e-07
   1e-04     0.003214172750     2.13e-08
   1e-05     0.003214172750     2.13e-08
   1e-06     0.003214168487     1.31e-06
   1e-07     0.003214069011     3.23e-05
   1e-08     0.003213784794     1.21e-04
   1e-09     0.003218758593     1.43e-03
```

The two methods agree to 8 significant figures at the best step sizes, with a relative error of 2.13e-08. Autograd is computing the true derivative, and there was no disagreement that needed explaining.

The more interesting result is that eps was swept rather than fixed, and the error is U-shaped rather than improving monotonically as eps shrinks. The two arms have different causes. At eps=1e-3 the secant spans enough of the loss curve to pick up its curvature, so the line through the two points is not the tangent at the middle; that error scales as eps squared, and the roughly 30x jump from 1e-4 is consistent with it. Below 1e-5 the opposite problem takes over: L(w+eps) and L(w-eps) agree in nearly all their digits, so subtracting them cancels the shared leading digits and leaves mostly rounding residue, which is then divided by 2*eps and amplified by roughly 1/eps. Each decade costs about an order of magnitude, which the table shows. The floor at 1e-4 and 1e-5 is where the two error sources cross.

The floor sits as low as 1e-08 only because the model is in float64. In fp32 the cancellation arm arrives several decades earlier and the best achievable agreement is a few digits rather than eight, which is the concrete reason gradient checks are conventionally done in double precision.

This verifies one element of one parameter tensor on one fixed batch with dropout off. That is strong evidence autograd is correct, not proof that it is correct everywhere.

## 3. Gradient accumulation, broken on purpose

Micro-batches of different lengths were accumulated two ways.

The correct reduction weights each micro-batch's mean loss by the number of tokens it contributed, so every token in the step carries equal weight. The broken reduction takes the average of the averages: each micro-batch's mean loss is divided by the number of micro-batches, so every micro-batch counts equally regardless of how many tokens it holds.

With `MICRO_LENS = [8, 8, 8, 64]` and B=8 throughout, the step holds 88 sequence positions worth of micro-batches. The broken reduction gives each 8-token micro-batch a full quarter of the gradient weight when it contributes only about 9 percent of the tokens, over-weighting it roughly 2.75x, and gives the 64-token micro-batch the same quarter when it holds 73 percent of the tokens, under-weighting it roughly 2.9x.

Both runs share the same seed, the same initialization and the same data at every step. The reduction is the only difference.

```
final loss  correct 2.9133   broken 2.9589
```

After 400 steps the broken run is 0.0456 worse. The direction is exactly what the weighting analysis predicts.

Interestingly nothing about the broken run looks broken. There is no divergence, no NaN, no gradient spike and no warning. The loss curve descends normally and would pass any smoke test. Plotted without the control curve beside it, the run looks entirely healthy.

Making the lengths equal collapses the gap to zero, because the two reductions then become the same arithmetic. The notebook leaves `MICRO_LENS` exposed so this can be checked directly.

## 4. Grad norm moving before the loss

At step 250 the model is fed a batch of uniformly random token ids, a sequence unlike anything in Shakespeare, so the gradient it produces is large and badly aimed.

The measurement is arranged so the lead is causal rather than an artifact of ordering. The logged loss is a fixed held-out probe batch, identical at every step, so it is not contaminated by which batch happened to be drawn. The probe is evaluated before the optimizer step, so the loss at step t reflects the model as it was going in. The grad norm at step t is measured on the gradient that is about to be applied. Harm from a bad gradient at step t therefore cannot appear in the loss until step t+1, by construction.

```
grad norm at 250: 5.716  (prev 0.562)
probe loss  249: 2.9876 | 250: 2.9853 | 251: 2.9984 | 252: 3.0119
```

At step 250 the grad norm jumped roughly 10x, from 0.562 to 5.716, which is z = +25.2 against the trailing window. The probe loss at that same step was 2.9853 and still falling, entirely unremarkable. One update later the loss reversed direction, rising 0.0131 at step 251 and another 0.0135 at step 252, giving back roughly two steps of progress.

A ranked scan of all 500 steps puts 250 first by a wide margin:

```
step 250 | grad z  +25.2 | loss z  -1.8 ->  -1.0 ->  -0.2
step 242 | grad z   +4.8 | loss z  -1.7 ->  -1.8 ->  -1.9
step 460 | grad z   +3.6 | loss z  -0.9 ->  -1.9 ->  -2.1
step 459 | grad z   +3.1 | loss z  -0.6 ->  -0.9 ->  -1.9
step 322 | grad z   +4.0 | loss z  -1.6 ->  -1.9 ->  -1.8
```

The gradient side of the signal is unambiguous, but the loss side never crossed a z of 2.5, with values of -1.0 and -0.2 in the two steps after the event. The loss response is real and plainly visible in the raw numbers, a trend reversal of about +0.027 over two steps against a run that was steadily declining, but it is small relative to step to step variation, so a z-test on the loss alone would not have flagged it.

The clean finding is the asymmetry. A 25-sigma event in the gradient signal produced a sub-2-sigma ripple in the loss, arriving a step later. Grad norm is the more sensitive instrument and it reports first. The other four ranked candidates have grad z between +3.1 and +4.8 with no coherent loss response, and are ordinary batch to batch noise rather than events.

## 5. MFU, measured and then explained

MFU is achieved FLOPs per second divided by hardware peak FLOPs per second. Achieved uses the standard transformer estimate of 6 * N * tokens per forward and backward pass, with N counting non-embedding parameters only.

```
non-embedding params : 788,736
tokens/sec           : 292,780
achieved             : 1.386 TFLOP/s
peak (Tesla T4 fp32) : 8.1 TFLOP/s
MFU                  : 17.11 %
```

The 6*N*tokens estimate counts matmul FLOPs only and ignores attention's quadratic term, which at block_size 128 is modest but not zero, so 17.11 percent is a floor rather than a point estimate. And the denominator is the fp32 peak, which is the honest choice because this run never touches a tensor core; it does mean the figure is not comparable to published MFU numbers that quote a bf16 or fp16 peak. Against the T4's 65 TFLOP/s fp16 peak the same throughput would read as roughly 2 percent, which is a different true statement about a different question.

Rather than guess at the cause, the same timing loop was re-run across widths and batch sizes with nothing else changed:

| n_embd | batch | non-emb params | tok/s | MFU % |
|---|---|---|---|---|
| 128 | 32 | 788,736 | 261,335 | 15.27 |
| 128 | 128 | 788,736 | 319,940 | 18.69 |
| 256 | 32 | 3,150,336 | 128,148 | 29.90 |
| 256 | 128 | 3,150,336 | 138,610 | 32.35 |
| 512 | 128 | 12,592,128 | 44,499 | 41.51 |
| 768 | 128 | 28,325,376 | 19,964 | 41.89 |

MFU rises from 15 percent to 42 percent purely by growing the shapes, and crosses 40 percent at n_embd 512 with no change to dtype, kernel or code path. Nothing was optimized between the first row and the last. Arithmetic intensity simply grew until the fixed per-step costs stopped dominating.

Width is the dominant factor. At batch 128, going from 128 to 256 buys 13.7 MFU points and 256 to 512 buys another 9.2. The matmuls get large enough to keep the SMs busy for a meaningful fraction of each kernel's lifetime. Batch size helps but far less: quadrupling from 32 to 128 buys 3.4 points at n_embd 128 and 2.5 points at n_embd 256. The curve then plateaus, with 512 to 768 buying only 0.4 points, which is within noise. That plateau near 42 percent is the practical fp32 roofline for this kernel mix on a T4. What remains beyond it is dtype, since tensor cores sit idle in fp32, plus attention's non-matmul work and memory-bound operations like LayerNorm and softmax that extra width does not fix.

Worth noting that tokens per second falls monotonically as the model grows while MFU rises. That is the expected relationship and it is the point of the metric: MFU measures how well the hardware is being used, not how fast the run finishes. The 768-wide model is the most efficient configuration in the table and the slowest to train.

One caveat on how clean the sweep is. `n_head` was held at 4 while n_embd varied, so head_dim grew from 32 to 192 alongside width. Width and head geometry therefore move together in this table rather than being varied independently. The direction of the result is not in doubt, but attributing the gain specifically to width would need a second sweep holding head_dim fixed.

So the original 17.11 percent is not a story about a badly written training loop. It is a 788k parameter model on a GPU built for much larger ones, and the same loop reaches 41.89 percent once the model is big enough to justify the hardware.

## 6. The number 0.1 in three formats

All three formats store sign, exponent and mantissa with an exponent bias. 0.1 is an infinite repeating fraction in binary, so none of them can hold it exactly and each rounds it differently.

```
fp32      0 01111011 10011001100110011001101   = 0.10000000149011612   err 1.490e-09
bf16      0 01111011 1001101                   = 0.10009765625000000   err 9.766e-05
fp8 E4M3  0 0011 101                           = 0.10156250000000000   err 1.562e-03

mantissa bits: fp32 23, bf16 7, E4M3 3
exponent bits: fp32 8 (bias 127), bf16 8 (bias 127), E4M3 4 (bias 7)
```

The fp8 E4M3 encoding is done by hand in the notebook rather than relying on a library type, since the point is to show the bits.

### Which one I would train in

bf16, and the bit layouts are the argument.

The exponent fields of fp32 and bf16 are byte identical for 0.1. Both are `01111011`, both use bias 127, both are 8 bits wide. bf16 is fp32 with sixteen mantissa bits removed and nothing else changed. The precision cost is visible and large, with representation error going from 1.490e-09 to 9.766e-05, about five orders of magnitude worse.

That is the right cost to pay, because the error is relative and training does not fail from relative error. It fails from range. Gradients span many orders of magnitude during training, and the failure mode that kills a run is small gradients flushing to zero, not a mantissa rounded in its seventh digit. bf16 gives up decimals and keeps the entire dynamic range of fp32, which is exactly the trade that matters. It is also why bf16 needs no loss scaling while fp16, which has the same 16 bits but only 5 exponent bits, does.

fp8 E4M3 is a different proposition. Its error on 0.1 is 1.562e-03, but the mantissa is not what rules it out. The binding constraint is the 4-bit exponent with bias 7, which cannot reach the magnitudes that late-training gradients occupy. Using it means per-tensor scaling factors and real engineering effort, which can be worth it at large scale for the memory bandwidth win, but it is not a drop-in choice.

What I would keep in fp32 regardless: the optimizer states and the master weight copy, since small updates repeatedly added to large weights are exactly where low precision silently rounds progress away; the loss reduction and any gradient accumulation, which section 3 above demonstrates the importance of; and the LayerNorm and softmax statistics. bf16 for the matmuls, fp32 for the bookkeeping, which is what mixed precision actually means.

## What each section is for

| section | what it shows |
|---|---|
| 1 | Every tensor in a step, with a line saying what each dimension means |
| 2 | Autograd agrees with finite differences to 8 significant figures, and why eps has a sweet spot |
| 3 | Average of averages over unequal micro-batches costs 0.0456 loss, silently |
| 4 | A 25-sigma gradient event produced a sub-2-sigma loss ripple one step later |
| 5 | 17.11 percent MFU is explained by model size, measured by sweep, and reaches 41.89 percent at n_embd 768 |
| 6 | 0.1 bit by bit in three formats, and the case for bf16 |

## Running it

The notebook is self contained. It downloads tiny-shakespeare on first run and needs only torch and matplotlib. It was run end to end on a Kaggle T4. To run it elsewhere, update the `PEAK` dictionary in section 5 with the peak FLOPs of the GPU actually in use, and set `PEAK_KEY` to match, since MFU is meaningless against the wrong denominator.
