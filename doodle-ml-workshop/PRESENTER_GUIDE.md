# Presenter guide — 60 minutes

Slides: https://docs.google.com/presentation/d/1eR-EpH2LtQqkN51yrW1VAq0jIObyY4FToDJKQyYFf5w/edit

This guide supplements the slide deck and fully implemented notebooks. The
slide deck is a first-pass visual draft and should be reviewed for final design
and speaker notes before the event.

## 0–5 | Hook: Draw → Guess → Laugh

Open the hosted demo or preloaded Colab playground. Ask a volunteer to draw
one of the six known classes. Make one easy prediction, then a weird or
ambiguous drawing. Ask: "Did it actually understand what you drew?"

Transition: We will figure out how it learned and build our own.

## 5–15 | Foundations + landscape

- Traditional code: programmer writes rules. ML: examples and labels are used
  to fit a decision function.
- Classical vs deep: engineered features + classifier vs neural networks that
  learn representations. Not a hard binary.
- Big landscape slide: show the breadth, then spotlight today's narrow path.
- Supervised classification vs unsupervised grouping.
- Classification predicts a category; regression predicts a number.
- Each 28×28 image has 784 pixel values.
- Training learns, validation compares choices, test checks once at the end.

Audience prompt: "If I shift the same drawing five pixels, is it still the
same object? Will the computer necessarily agree?"

## 15–20 | Partner game 1: Become the Classifier

Show 3–4 doodles (house, fish, tree, bicycle). Each pair has 90 seconds to
write two yes/no rules to classify them. Partners exchange rules and attempt
to find an ambiguous drawing. Reveal: hand-written questions resemble a tree;
choosing the closest examples resembles KNN.

## 20–28 | Method 1: Similarity & Rules

Run method_1_similarity_rules.ipynb.
- KNN: inspect the nearest neighbors for one drawing.
- Change k, show validation trend.
- Decision Tree: show the top three levels, then change depth.
- Ask: "Which approach has a rule you can explain? Which remembers examples?"

Keep the notebook code prewritten; no live coding.

## 28–36 | Method 2: Features & Forests

Run method_2_forests_features.ipynb.
- Explain forest votes; show first 12 individual tree votes.
- Show same doodle before and after centering.
- Compare raw pixels and HOG visually.
- Run validation comparison; don't predict a winner in advance.

## 36–41 | Partner game 2: Fool the Classifier

Ask pairs to predict a failure mode, then draw an unusual version of one
allowed class: off-center, sparse, stylized, or overlapping.
Collect 1–2 examples for the live demo. Ask whether the problem comes from
the model, representation, or training distribution.

## 41–48 | Method 3: SVM & Evaluation

Run method_3_svm_evaluation.ipynb.
- SVM: wide separating margin in a feature space.
- HOG + linear SVM as a classical computer vision pipeline.
- Read confusion matrix by row and column.
- Compare validation results, with each pipeline's features clearly named.
- Show mistakes. Test split stays sealed.

## 48–53 | Two advanced extensions

- Active learning: which uncertain doodle should we ask a human to label?
- Unknown classes: a dinosaur may be forced into one of six known labels.
- No false claim that current system detects unknown objects reliably.

## 53–60 | Final Boss: build a baseline

Open final_challenge.ipynb, already shared at minute zero.
- 1 minute: select KNN baseline; run train cell.
- 2 minutes: inspect validation accuracy/confusion matrix.
- 2 minutes: copy an alternative from Method 2 or 3.
- 2 minutes: compare and choose. Students can run the Gradio canvas after
  the session or if time allows.

Bronze = working baseline; Silver = valid comparison; Gold = documented
error analysis and improvement. Do not announce accuracy thresholds without
real measured benchmark results.

## After workshop | Extension

Run the sealed test exactly once, launch personal Gradio app in Colab,
swap doodles with a partner, and document what fails.

## Emergency fallbacks

- Dataset access fails: use organizer's prepared cached .npz file.
- Colab startup slow: show pre-run notebook cells/screenshots.
- Gradio share fails: inspect validation images and predictions in notebook.
- Too little time: skip cross-validation demo and advanced extension.

## Presenter technical checklist

- [ ] Run all four notebooks from a clean Colab environment.
- [ ] Check Gradio canvas and prediction orientation.
- [ ] Confirm category names, counts and preprocessing.
- [ ] Run measured validation benchmark; update scoreboard.
- [ ] Prepare a backup demo and test it on projector.
- [ ] Verify slides are visually polished and note placeholders removed.
