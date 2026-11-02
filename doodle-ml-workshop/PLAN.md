# Workshop implementation plan

## Learning outcome
Within 60 minutes, a beginner with basic Python can explain supervised
classification, run four classical models, inspect failures, and adapt working
notebook code to build a doodle recognizer.

## Session plan
| Time | Activity |
| --- | --- |
| 0–5 | Live Gradio doodle demo, prediction and surprising mistake |
| 5–15 | Classical ML landscape, features/labels, splits and roadmap |
| 15–20 | Partner game: Become the Classifier |
| 20–28 | Method 1: KNN vs Decision Trees |
| 28–36 | Method 2: Random Forests + features |
| 36–41 | Partner game: Fool the Classifier |
| 41–48 | Method 3: SVM + confusion matrix |
| 48–53 | Active learning and unknown-class limitations |
| 53–60 | Final Challenge: guided first baseline |
| After | Optional 15–30 min model improvement and fresh drawings |

## Presenter prep — mandatory
1. Run the entire notebook sequence in fresh Colab runtimes.
2. Run python -m scripts.prepare_dataset to prefetch official sample.
3. Run python -m scripts.benchmark and record REAL validation results.
4. Calibrate Bronze/Silver/Gold milestones from those measured results.
5. Train and host a trusted demo model, or demo from a prepared Colab runtime.
6. Test Gradio sketch orientation (black pen on white canvas) and preprocessing.
7. Keep a saved model and screenshots as a backup if network access fails.
8. Share the four Colab notebook links before the workshop starts.

## Pedagogical constraints
- All three Method notebooks are fully implemented; students run/tweak.
- Only final_challenge.ipynb has TODO prompts, with a runnable baseline.
- Do not select a model based on test results.
- Comparing different features is okay if each is labeled explicitly.
- A 28x28 image is 784 pixel values, not a semantic understanding of a cat.
- LinearSVC decision_function scores are margins, not calibrated probabilities.
- Quick Draw subset uses first 900 rows/category; not a global random benchmark.
- Do not claim the hosted Space uses student-trained models.

## Acceptance criteria
- Four notebooks load and run in clean Colab.
- Dataset is reproducible with a fixed random seed.
- No code path uses y_test for tuning.
- The Gradio drawing conversion produces a 1x784 white-ink image.
- Model training and drawing use the same feature transformations.
- Notebook titles, README links, and slide sections agree.
- All slides contain verified claims or clearly illustrative examples.
- Measured model scores and tested demo URL replace placeholders before event.

## Contingencies
- Internet fails: use preloaded notebook runtime and backup screenshots.
- Gradio share link fails: show predictions on held-out examples in notebook.
- Slow devices: run the baseline only; partner students.
- Short session: make the final challenge a 7-minute guided build with extensions.

