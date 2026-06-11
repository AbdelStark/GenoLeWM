# What does this edit do?

*I tried to build a latent world model for genomic edits. It failed. The failure is the useful part.*

A DNA language model is mostly a reader.

Give it a stretch of genome and it can tell you something about that sequence: how likely it looks, what context it resembles, where it seems surprising, which parts fit the model's learned distribution of DNA.

That is useful, but it is not the question I wanted GenoLeWM to answer.

The question was:

Given a DNA window and a specific edit, what changes?

Not the letters. The letters are the easy part. If you change an A to a T, insert three bases, or delete a codon, the edited sequence is deterministic.

The interesting question is what changes inside the model.

Does the edit move the model's representation of the sequence? Does it move it a little or a lot? Does a chain of edits move it in a way we can predict without rerunning the big model each time?

That is the world-model version of variant reasoning.

A world model takes a state and an action, then predicts the next state. In this project, the state is a frozen DNA model's embedding of a genomic window. The action is a genomic edit. The next state is the embedding of the edited window.

If this worked, it would be useful.

You could encode a reference window once, then score many edits cheaply. You could roll out a multi-edit haplotype as a sequence of latent transitions. You could search over edits in latent space and ask which actions move a sequence toward a target representation.

That was the bet behind GenoLeWM.

The short version of the result is simple: the bet did not pay off in this release.

The trained predictor did not beat Carbon's own zero-shot baseline on the main released variant-effect slices. The planning demo showed that the pipeline runs, not that it plans useful edits. The K=20 rollout-speed target is still open. Most importantly, the learned rollout was much worse than the stupidest possible baseline: predict that nothing changed.

That sounds bad because it is bad.

It is also the most interesting thing the project found.

## The setup

GenoLeWM uses Carbon-500M as the DNA encoder. Carbon is the heavy model. It reads the reference DNA window and produces an embedding, a 1,024-dimensional vector that summarizes the region around the edit.

For the default window size in this project, 12,288 base pairs become 2,048 DNA tokens because Carbon tokenizes DNA in 6-base chunks.

Call the reference embedding:

```text
s_t
```

Now take an edit: position, edit type, reference bases, alternate bases. Pass that edit through a smaller action encoder. Then feed the state and the edit embedding into a trainable predictor.

The predictor tries to output:

```text
ŝ_{t+1}
```

That is the predicted embedding after the edit.

To train it, you also do the expensive thing for real: apply the edit to the DNA window, run Carbon again, and get the actual target embedding:

```text
s_{t+1}
```

The training target is not the edited DNA string. The target is Carbon's representation of that edited string.

In code-shaped form:

```text
s_t       = Carbon(reference window)      # frozen
a         = action_encoder(edit)          # trainable
ŝ_{t+1}   = predictor(s_t, a)             # trainable
s_{t+1}   = Carbon(edited window)         # frozen target

loss      = direction error + small magnitude error
```

More concretely, the prediction loss combines cosine distance with a scaled MSE term:

```text
loss = (1 - cos(ŝ_{t+1}, s_{t+1})) + 0.1 * MSE(ŝ_{t+1}, s_{t+1}) / d
```

This is a Joint-Embedding Predictive Architecture applied to genomic edits.

The point of a JEPA is that you predict in representation space instead of reconstructing raw input. You do not generate DNA letters. You predict where the edit lands in the encoder's latent space.

That choice is attractive for three reasons.

First, the target is compact. A 12kb DNA window becomes one vector.

Second, the output space has geometry. You can measure angles, distances, residuals, and rollout drift.

Third, the economics are tempting. Carbon is the expensive part. If you can pay for Carbon once, cache the reference state, and then do many cheap predictor calls, you get a clean path to fast variant scoring and latent planning.

The slogan was:

> Pay for the encoder once. Query it many times.

That is a good slogan. It is also where the trap begins.

## Why freezing looked like the right move

Freezing Carbon had two obvious advantages.

The first was compute.

Carbon-500M dominates memory and runtime. The action encoder and predictor are much smaller. If Carbon stays frozen, the system can be designed around cached embeddings and cheap downstream inference.

The second was stability.

JEPAs can collapse when the encoder and predictor are trained together. The encoder can learn to map many different inputs to nearly the same vector, making the prediction task artificially easy and the representation useless.

With a frozen encoder, that specific failure mode is closed. Carbon's targets are fixed. The predictor cannot drag the encoder into collapse.

At the start, freezing looked like a bargain: cheaper inference, simpler training, fixed targets.

The project's main lesson is that the bargain has another side.

Freezing also fixes the geometry.

If the effect of an edit is not already visible in Carbon's latent space, the predictor cannot make it visible. It can only learn inside the geometry it was given.

That distinction ended up mattering more than the compute story.

## What happened on the released benchmarks

The fair question was never whether GenoLeWM was a clinical model. It is not.

The fair question was narrower:

Does the learned edit predictor add useful signal beyond Carbon's own zero-shot likelihood scoring?

On the released v0.2 slices, the answer is no.

Higher is better in the table below. The gap is GenoLeWM minus Carbon zero-shot.

| Benchmark                  |  N | GenoLeWM | Carbon zero-shot |     Gap |
| -------------------------- | -: | -------: | ---------------: | ------: |
| ClinVar coding, AUROC      | 16 |   0.7344 |           0.9219 | -0.1875 |
| ClinVar non-coding, AUROC  | 16 |   0.5625 |           0.8750 | -0.3125 |
| BRCA2 saturation, Spearman | 32 |   0.1492 |           0.4769 | -0.3277 |

Read that table with the right level of seriousness.

These slices are tiny. Sixteen variants for each ClinVar slice and thirty-two for BRCA2 are not enough for a population-scale model-quality claim.

But they are enough to block a positive claim.

There were a couple of favorable deltas elsewhere, including coding ClinVar balanced accuracy and a small positive delta on the TraitGym Mendelian slice because Carbon's own correlation was more negative there. Those do not rescue the story. The main ranking metrics trail the baseline, non-coding is worse, BRCA2 is worse, and the overall evidence does not support "GenoLeWM improves on Carbon."

A weak model can still have a strong systems contribution. A benchmark harness, release pipeline, and artifact story can be real even when the model result is negative.

But we should not confuse those things.

GenoLeWM has credible release plumbing. It does not yet have a credible model-quality win.

## The result that explains the result

The variant-effect table is not the most important result.

The rollout result is.

Rollout fidelity asks a more direct world-model question:

If I apply a chain of edits in latent space, does the predictor land near Carbon's embedding of the actually edited sequence?

For each edit chain, we compare the predicted final embedding to the true final embedding using cosine similarity. A cosine near 1 means the vectors point in almost the same direction. Lower is worse.

The predictor scored:

| Rollout condition     |  N | GenoLeWM cosine | Source-state baseline |
| --------------------- | -: | --------------: | --------------------: |
| Phased haplotypes     |  8 |           0.289 |                 0.998 |
| Synthetic edit chains |  8 |           0.302 |                 0.991 |

The source-state baseline is brutally simple.

It ignores the edit and returns the original embedding:

```text
prediction = s_t
```

In other words, it says: nothing changed.

That baseline was almost perfect. The learned model was much worse.

This is the result I kept coming back to, because it explains almost everything else.

A world model that loses to "nothing happened" has not learned the transition. It has learned some transformation of the state that is less faithful than leaving the state alone.

Once you see why, the failure becomes less mysterious and more useful.

## The latent-residual trap

Imagine a 12,288 base-pair window.

Now change one base.

From the point of view of the raw sequence, that edit may matter a lot biologically. It could hit a splice site, alter a codon, disrupt a motif, or do nothing. Biology can be sharp.

But from the point of view of a pooled 1,024-dimensional embedding of the whole window, the edit is tiny.

Most of the input did not change. The model reads almost the same sequence before and after the edit. The true target embedding after the edit, `s_{t+1}`, is usually very close to the original embedding, `s_t`.

So the learning problem has a strange shape.

The predictor is asked to predict the full next embedding:

```text
s_{t+1}
```

But the useful edit effect is only the residual:

```text
Δ = s_{t+1} - s_t
```

And that residual is small.

The full target is dominated by the part of the sequence that did not change. The loss mostly rewards getting the big unchanged component right. Copying `s_t` already gives you a very high cosine score.

That is the latent-residual trap.

You want to learn the edit effect, but the training target mostly says: preserve the input.

This gets worse because the predictor was initialized to behave like an identity function. The final output layer was zero-initialized so the model starts near `s_t`. That is a sensible residual-network trick. It keeps early training stable.

But in this setting, the initialization places the model directly on the do-nothing baseline.

Training has to push the model away from a very strong copy solution to learn a tiny residual that Carbon may not make easy to read out.

That is a hard ask.

The released model did not learn a clean residual. It learned a distorted copy. And a distorted copy is worse than a clean copy.

That explains the rollout result. It also explains the weak variant rankings.

If your score depends on how much the predicted state moved, and the predicted movement is mostly model artifact, then your variant ranking is not measuring edit effect. It is measuring predictor distortion.

## This is not just a genomics lesson

The broader lesson is about frozen foundation-model dynamics.

A lot of latent world-model ideas have the same shape:

1. Take a powerful pretrained encoder.
2. Freeze it.
3. Train a small action-conditioned model on top.
4. Predict next embeddings instead of next observations.
5. Use the small model for rollout, planning, or cheap scoring.

That recipe is attractive. It often sounds more efficient and more stable than end-to-end training.

But it only works if the action is visible in the frozen representation.

That is the core test.

Not "is the encoder good?"

Not "is the predictor expressive?"

Not "is the architecture elegant?"

The first question is:

> Does the frozen embedding actually expose the action effect at a scale the predictor can learn?

The second question is:

> Does the learned predictor beat the no-change baseline?

If the answer to either question is no, the world-model framing is probably ahead of the evidence.

Freezing solves one problem and creates another. It closes the encoder-collapse route, but it also removes the encoder's ability to reorganize its space around edit predictability.

You cannot get both full freezing efficiency and full representation adaptation in the same move.

That is not a minor implementation detail. It is the central tradeoff.

## The efficiency story is still unproven

There is another result that should be stated carefully.

The released single-variant scoring path measured about 115 seconds per variant.

That sounds disastrous, but it is not the steady-state regime GenoLeWM was designed for. That number includes cold startup costs, model loading, and repeated Carbon calls. It is not a clean measurement of the amortized latent-only path.

The intended regime is different:

1. Carbon already loaded.
2. Reference embeddings cached.
3. Many edits scored in the same region.
4. Predictor calls dominate.
5. Carbon is not rerun for every hypothetical rollout step.

That is the architecture GenoLeWM was meant to make possible.

But the release did not fully demonstrate that end-to-end regime. So the honest conclusion is not "GenoLeWM is fast" and not "GenoLeWM is slow."

The honest conclusion is:

> The efficiency thesis is architecturally plausible and operationally unproven.

One speed result did move in the right direction. KV-cached autoregressive rollout gave about 2.41x speedup at K=5 and 2.47x at K=20 compared with the naive rollout path.

That is useful engineering evidence.

It is not closure.

K=20 still missed the 5x target, and the broader amortized scoring story still needs a real benchmark on the intended path.

This is exactly where credibility gets won or lost. Do not sell the benchmark you wanted to measure. Report the benchmark you actually measured.

## The part I am most willing to defend

The model result is negative.

The release discipline is the part of the project I like.

Every reported result is tied to artifacts. The model has a content-addressed identity. The generated paper is rendered from machine-readable benchmark and planning files. The release package preserves the negative findings instead of letting them drift into a success story.

The receipt system is deliberately narrow. It proves checksum provenance and artifact identity. It does not prove privacy, runtime behavior, clinical safety, or scientific truth.

That boundary matters.

Good infrastructure does not make a weak model strong. It makes the weakness inspectable.

That is worth something.

A negative result is only useful if people can reproduce it, audit it, and see exactly where the claim stops.

GenoLeWM's best current claim is not "we built a better variant predictor."

It is:

> We built and released an artifact-bound genomic edit world-model pipeline, and the measured model did not beat the baseline.

That is less exciting than the original goal.

It is also true.

## What I would test next

The next round should not start by training a larger predictor.

It should start by testing whether the residual is learnable at all.

### 1. Measure the residual directly

Before another big training run, measure the distribution of:

```text
||s_{t+1} - s_t||
cos(s_t, s_{t+1})
```

Break it down by edit type, genomic context, conservation, coding consequence, splice proximity, and distance from the pooling center.

If most edits barely move the embedding, the no-change baseline will remain hard to beat.

That is not a training bug. That is the geometry of the target.

### 2. Probe the frozen space before building dynamics

Train simple probes to predict:

```text
Δ = s_{t+1} - s_t
```

Start with a linear map. Then try a small MLP. Compare both against nearest-neighbor residual retrieval from similar contexts.

If a linear probe works, the signal is present and the predictor or loss is probably the problem.

If an MLP works but the linear probe fails, the signal is present but not simple.

If neither works, the frozen encoder probably does not expose edit effects in a useful form.

That experiment is cheaper and more informative than scaling the current architecture.

### 3. Stop training only on the full state

The full-state target makes copying too rewarding.

Future variants should try residual-aware objectives: predict `Δ` directly, normalize residuals, reweight examples by residual size, or use a loss that forces the model to explain movement beyond the source-state baseline.

The target should make the edit effect loud enough to learn.

### 4. Let the encoder move, but only carefully

A near-frozen encoder may be the right compromise.

LoRA or another lightweight adapter could let Carbon bend its representation enough to make edit effects more legible, while a regularizer and collapse diagnostics keep the representation from degenerating.

This gives back some adaptation while preserving most of the pretrained model.

It also reopens the collapse risk that freezing avoided.

That is the tradeoff. It should be managed explicitly, not wished away.

### 5. Start with loud edits

Single SNVs are often the quietest possible training signal in a global window embedding.

Start with edits that should move the representation more: nonsense mutations, splice-disrupting edits, motif-breaking regulatory edits, larger indels, or edits in highly constrained regions.

If the model cannot learn the loud cases, the quiet cases are not going to save it.

## What this project rules out

It does not rule out genomic edit world models.

It does not rule out JEPA-style prediction over DNA embeddings.

It does not prove that Carbon's latent space cannot support edit-conditioned dynamics.

It rules out the easiest version of the story:

> Freeze Carbon, train a small predictor on full next-state embeddings, and expect useful edit dynamics to appear.

That version failed on the released evidence.

The failure mode is clean enough to be useful. The edit residual is small. The frozen geometry was not trained to expose it. The identity baseline is extremely strong. The predictor learned a worse copy instead of a meaningful transition.

That is the result.

## Takeaway

A genomic edit world model is a good idea.

The current GenoLeWM release is not evidence that the idea works.

It is evidence that the naive frozen-encoder version can fail in a very specific way.

The lesson is portable:

> Frozen-feature world models live or die on whether the action is already visible in the frozen geometry.

Before training the dynamics model, test the geometry.

Before claiming rollout, beat the no-change baseline.

Before claiming efficiency, measure the amortized path.

Before claiming model quality, beat the encoder's own baseline on enough data to matter.

I wanted GenoLeWM to work. It did not.

The useful thing is that it failed in a way we can inspect, reproduce, and learn from.

That is not the result I wanted, but it is a result worth keeping.

*GenoLeWM is an alpha research system. It is not a clinical or diagnostic tool, not medical evidence, and not a deployment or privacy-assurance claim.*

[1]: https://github.com/AbdelStark/GenoLeWM "GitHub - AbdelStark/GenoLeWM: An action-conditioned JEPA world model for DNA, built on top of Carbon. · GitHub"
[2]: https://huggingface.co/HuggingFaceBio/Carbon-500M "HuggingFaceBio/Carbon-500M · Hugging Face"
[3]: https://raw.githubusercontent.com/AbdelStark/GenoLeWM/main/README.md "raw.githubusercontent.com"
[4]: https://raw.githubusercontent.com/AbdelStark/GenoLeWM/main/ARCHITECTURE.md "raw.githubusercontent.com"
[5]: https://raw.githubusercontent.com/AbdelStark/GenoLeWM/main/PRIVACY.md "raw.githubusercontent.com"
