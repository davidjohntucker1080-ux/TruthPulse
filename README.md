# Tell

Tell is a two-player lie detection game and a real experiment. It asks one question: **can a computer reading tiny face movements spot lies better than chance on people it has never seen?**

Play it on a phone: open the GitHub Pages link in Safari or Chrome, then tap Share → Add to Home Screen.

## How a game works

1. One player answers and one player guesses.
2. Before each question, the answerer privately sees **TRUTH** or **LIE**. Each game has the same number of each, in random order.
3. The answerer reads the question out loud and answers it while the front camera tracks their face.
4. The guesser makes a call, then Tell reveals the answer.
5. Once Tell has learned from other people, it makes its own guess after the human does.

Because the app assigned every truth and lie, we always know the right answer. This "ground truth" is what most real-world lie detection data lacks.

## What gets measured

Tell uses [MediaPipe Face Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker), which runs entirely on the phone. About 30 times a second, while the person answers, Tell records:

- 52 facial movement scores, such as blinks, eye direction, brow raises, lip presses, smiles and jaw movement
- Head angles: turning, nodding and tilting

For each answer it then works out:

- The average and variability of each movement
- Blink rate
- How much and how fast the head moves
- How long the answer took
- How much of the time the face was out of view

**No video or sound is ever saved.** Everything stays on the phone in browser storage until someone exports it.

## How the analysis stays honest

- **Each person is compared with themselves.** Every measurement is scaled against that person's own average across all their answers, so someone who naturally blinks a lot doesn't look like a liar. This step never uses the truth/lie labels.
- **The model is tested on people it hasn't seen.** It's trained on everyone except one person, tested on that person, and this repeats for each person (leave-one-person-out). Testing on the same people a model learned from is the main reason past AI lie detectors looked better than they were.
- **Results are compared with chance.** Accuracy is checked against a coin flip (50%) with a one-sided binomial test.
- **Single measurements are leads, not findings.** The lab page lists the measurements that differ most between lies and truths. With more than 100 measurements, some will differ by luck, so these are things to test, not results.

The model is L2-regularized logistic regression. It's simple on purpose: with small datasets, simple models are harder to fool yourself with. The code is in `analysis.js`.

## Collecting data from many people

Each phone keeps its own data. To combine it:

1. On each phone, go to **Your data → Export data**. This saves a `.json` file.
2. Send the files to one phone or computer.
3. On that device, go to **Your data → Import data** and choose the files.

The lab results then cover everyone. Names are only used to group each person's answers, so use the same name for the same person on every phone.

A trustworthy result needs a few hundred answers from at least 10 to 20 different people.

## Files

| File | What it does |
| --- | --- |
| `index.html`, `style.css` | Screens and styling |
| `app.js` | Camera, face tracking, game flow, live readings, lab page |
| `analysis.js` | Feature extraction, per-person normalization, model, statistics |
| `questions.js` | Question bank |
| `db.js` | On-device storage (IndexedDB) |
| `vendor/mediapipe/` | MediaPipe Tasks Vision 0.10.14 (Apache-2.0), bundled so the app needs no outside services |
| `models/face_landmarker.task` | MediaPipe face landmark model (Apache-2.0) |

## Running it locally

The camera only works over HTTPS or on `localhost`:

```sh
python3 -m http.server 8000
# open http://localhost:8000
```

## Limits to keep in mind

- Lies told in a game are low stakes. The signs of real lies may be stronger, weaker or different.
- Results from friends and family may not apply to everyone.
- Even a model that beats chance won't be right every time. It should never be used to accuse anyone of lying.
