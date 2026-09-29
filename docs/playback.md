# New to Savante

A short guide to Savante's page: what you see, how to ask her something, and how to listen to her. For installing and
running bankml, see [usage.md](usage.md).

**Savante** is an agent whose identity is written down and checked: her canon (who she is, her oath, what she
believes) is hashed into a ledger, and the page refuses to speak as her if it does not match. She runs on your own
computer, on a model bankml has verified. Her answers are drafts, never verdicts.

## Open her page

After `./install.sh`, open **http://127.0.0.1:7873**. The page only works on this computer.

- **The chat** is in the middle. Type a question and press **Send**.
- **The side panel** holds her portrait and the settings. Drag its grip (**⠿ panel**) to put it on the other side, or
  press **⇄**. Drag the small corner handles to resize. The page remembers your layout.

## Your first question

1. Type something short, like *"Who are you?"*, and press **Send**.
2. A timer starts. It first says *reading the prompt*: the model is reading Savante's instructions. On a laptop this
   takes about **2 minutes** the first time. Then it says *writing*, and the answer appears a few words at a time.
3. Under the answer you see when you sent it, how long it took, and a **receipt**: proof that this machine's verified
   model wrote it.

**Stop** cancels an answer. **New session** starts over. Your last conversation comes back when you reopen the page.

Want her judgement on something? Start with *review:*, e.g. *"review: is this plan sound?"*. She answers in a fixed
form: findings, verdict, reasons, conditions and risks. Treat it as a draft.

## Her card

Click her **portrait** to open her card: her name, mantra and description, her oath and beliefs, and the checks that
prove this is really her (each file's fingerprint and whether it matches the ledger). Close it with **✕** or by
clicking outside it.

## Listening to her

Savante speaks from her card.

| press | what you hear |
|---|---|
| **▶ PLAY** beside her name | who she is, in a few sentences |
| **▶ PLAY THE INTRODUCTION** | eight short chapters: who she is, her oath, what she believes, why she exists, and more |
| **▶ PLAY THE READING** | bankml's idea, read from its technical report: one-bit and three-valued (ternary) model weights, and how an ordinary CPU computes with them |
| **▶** beside a chapter or a sentence | plays from there |

While she plays, the button becomes **❚❚ PAUSE**, then **▶ RESUME**. **■ STOP** ends it. The introduction picks up
where you left off.

**The voice dial.** When she starts speaking, a small set of knobs appears beside the card:

- **SPEED**: slower or faster, without changing her pitch.
- **GAIN** and **VOLUME**: how loud.
- **FM RATE** and **FM DEPTH**: a wobble effect; leave them at 0 for her natural voice.

Drag the dial by **⠿** to move it. Your settings are remembered.

**Keep a copy.** **⤓ Savante.opus** downloads the introduction and her voice examples as one audio file, and
**⤓ Savante-reading.opus** the reading. Each has a chapter mark per chapter.

## Letting others listen

If you start the network page (`./install.sh start --view`), people on your network can open
`http://<your computer's address>:7874` and play the same introduction and reading, with the same dial. They can
listen and watch, but not chat, and they never see your conversations.

## Her voice on your computer

Her recordings come with bankml, so she speaks right away. To let her say new things (after you change her texts),
install her voice engine once:

```sh
./install.sh voice      # Piper and the en_GB-cori-high voice, no sudo
python3 sAGI/speak.py     # render every clip; run one at a time on a small machine
```

Without it, a simpler stand-in voice is used. Her voice is her own, built from open parts: Piper's
`en_GB-cori-high` (public-domain recordings), pitched to 182 Hz. Her name is said *sav-ont*.

## The other tabs, briefly

| tab | what it holds |
|---|---|
| **.history** | every question and answer, newest first, with a search bar |
| **Responses** | one answer at a time; copy it, save it as a note, or get a proof that it is in your history |
| **.memory** | your own notes; Savante reads them as your notes, not as evidence |
| **Metrics** | how fast she answers, over time |
| **Models** | add a model or switch to another ([usage.md §5](usage.md#5-models)) |
| **Integrity** and **Verifier** | re-check that she is who her ledger says |

More detail on any of these: [usage.md](usage.md), from §8 on.
