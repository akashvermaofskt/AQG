# Automatic fill in the blanks

## How to set up the project environemnt

* [Clone](https://github.com/akashvermaofskt/AQG.git) this repo in your local machine.
* [Install](https://blog.ruanbekker.com/blog/2018/11/27/python-flask-tutorial-series-create-a-hello-world-app-p1/) python3. 
* cd ⁨`AQG`
* setup [virtual environment](https://docs.python.org/3/library/venv.html) in it.
* Install all required packages mentioned in `requirement.txt`.

## How to run

* Open terminal then `cd AQG` 
* Run command `python3 api.py` in your terminal


## How to hit APIs

#### To Generate Summary

* Hit following API http://0.0.0.0:5000/summary i.e `GET`

##### Body

```
{
  "Input": {
    "Text": "An old man lived with his four sons in a village. The old man was worried. His sons were always quarrelling with each other. He had tried telling them many times to avoid fighting. But his sons would not listen to him.One day, he called his four sons. He gave them a small bundle of sticks, and asked them to break the bundle into two. The bundle was made of four sticks. “It’s child’s play,” said the eldest son. He took the bundle and tried to break it. He was surprised that the sticks in the bundle remained intact. He used more force. He tried again and again. He started panting for breath. The bundle would not break. He gave up.Then his brothers tried to break the bundle of four sticks without success.Their father smiled and asked them to untie the bundle. He asked each brother to take one stick and try to break it. Each of the sons took a stick in hand. In no time, the sticks were bent and broken.“A single stick is easily broken. If four sticks come together it is impossible to break them,” said the old man, giving his sons a meaningful look.This time the lesson went home. The brothers stopped fighting each other. They would work together as a team and succeed in doing whatever work was given to them.The four boys had discovered that unity is strength.",
    "Lines": 10
  }
}
```


* This will return a JSON file in this format.

```
{
  "Output": {
    "Summary": [
      "An old man lived with his four sons in a village.",
      "But his sons would not listen to him.One day he called his four sons.",
      "He gave them a small bundle of sticks and asked them to break the bundle into two.",
      "The bundle was made of four sticks.",
      "He took the bundle and tried to break it.",
      "He was surprised that the sticks in the bundle remained intact.",
      "The bundle would not break.",
      "He gave up.Then his brothers tried to break the bundle of four sticks without success.Their father smiled and asked them to untie the bundle.",
      "Each of the sons took a stick in hand.",
      "If four sticks come together it is impossible to break them said the old man giving his sons a meaningful look.This time the lesson went home."
    ]
  }
}
```

 #### To generate Fill in the blanks

* Hit following API http://0.0.0.0:5000/answer_list i.e `GET`

##### Body

```
{
  "Input": {
    "Summary": [
      "An old man lived with his four sons in a village.",
      "His sons were always quarrelling with each other.",
      "But his sons would not listen to him.One day he called his four sons.",
      "He gave them a small bundle of sticks and asked them to break the bundle into two.",
      "The bundle was made of four sticks.",
      "He took the bundle and tried to break it.",
      "He was surprised that the sticks in the bundle remained intact.",
      "The bundle would not break.",
      "He gave up.Then his brothers tried to break the bundle of four sticks without success.Their father smiled and asked them to untie the bundle.",
      "Each of the sons took a stick in hand.",
      "If four sticks come together it is impossible to break them said the old man giving his sons a meaningful look.This time the lesson went home."
    ]
  }
}
```

* This will return a JSON file in this format.

```
{
  "Number_of_question": 25,
  "QAs": [
    {
      "Answer": "old man",
      "Original_Sentence": "An old man lived with his four sons in a village.",
      "Question": "An _______ lived with his four sons in a village."
    },
    {
      "Answer": "four",
      "Original_Sentence": "An old man lived with his four sons in a village.",
      "Question": "An old man lived with his _______ sons in a village."
    },
    {
      "Answer": "village",
      "Original_Sentence": "An old man lived with his four sons in a village.",
      "Question": "An old man lived with his four sons in a _______."
    },
    {
      "Answer": "day",
      "Original_Sentence": "But his sons would not listen to him.One day he called his four sons.",
      "Question": "But his sons would not listen to him.One _______ he called his four sons."
    },
    {
      "Answer": "four",
      "Original_Sentence": "But his sons would not listen to him.One day he called his four sons.",
      "Question": "But his sons would not listen to him.One day he called his _______ sons."
    },
    {
      "Answer": "small bundle",
      "Original_Sentence": "He gave them a small bundle of sticks and asked them to break the bundle into two.",
      "Question": "He gave them a _______ of sticks and asked them to break the bundle into two."
    },
    {
      "Answer": "bundle",
      "Original_Sentence": "He gave them a small bundle of sticks and asked them to break the bundle into two.",
      "Question": "He gave them a small _______ of sticks and asked them to break the _______ into two."
    },
    {
      "Answer": "two",
      "Original_Sentence": "He gave them a small bundle of sticks and asked them to break the bundle into two.",
      "Question": "He gave them a small bundle of sticks and asked them to break the bundle into _______."
    },
    {
      "Answer": "bundle",
      "Original_Sentence": "The bundle was made of four sticks.",
      "Question": "The _______ was made of four sticks."
    },
    {
      "Answer": "four",
      "Original_Sentence": "The bundle was made of four sticks.",
      "Question": "The bundle was made of _______ sticks."
    },
    {
      "Answer": "bundle",
      "Original_Sentence": "He took the bundle and tried to break it.",
      "Question": "He took the _______ and tried to break it."
    },
    {
      "Answer": "bundle",
      "Original_Sentence": "He was surprised that the sticks in the bundle remained intact.",
      "Question": "He was surprised that the sticks in the _______ remained intact."
    },
    {
      "Answer": "bundle",
      "Original_Sentence": "The bundle would not break.",
      "Question": "The _______ would not break."
    },
    {
      "Answer": "bundle",
      "Original_Sentence": "He gave up.Then his brothers tried to break the bundle of four sticks without success.Their father smiled and asked them to untie the bundle.",
      "Question": "He gave up.Then his brothers tried to break the _______ of four sticks without success.Their father smiled and asked them to untie the _______."
    },
    {
      "Answer": "four",
      "Original_Sentence": "He gave up.Then his brothers tried to break the bundle of four sticks without success.Their father smiled and asked them to untie the bundle.",
      "Question": "He gave up.Then his brothers tried to break the bundle of _______ sticks without success.Their father smiled and asked them to untie the bundle."
    },
    {
      "Answer": "success.Their father",
      "Original_Sentence": "He gave up.Then his brothers tried to break the bundle of four sticks without success.Their father smiled and asked them to untie the bundle.",
      "Question": "He gave up.Then his brothers tried to break the bundle of four sticks without _______ smiled and asked them to untie the bundle."
    },
    {
      "Answer": "bundle",
      "Original_Sentence": "He gave up.Then his brothers tried to break the bundle of four sticks without success.Their father smiled and asked them to untie the bundle.",
      "Question": "He gave up.Then his brothers tried to break the _______ of four sticks without success.Their father smiled and asked them to untie the _______."
    },
    {
      "Answer": "stick",
      "Original_Sentence": "Each of the sons took a stick in hand.",
      "Question": "Each of the sons took a _______ in hand."
    },
    {
      "Answer": "hand",
      "Original_Sentence": "Each of the sons took a stick in hand.",
      "Question": "Each of the sons took a stick in _______."
    },
    {
      "Answer": "four",
      "Original_Sentence": "If four sticks come together it is impossible to break them said the old man giving his sons a meaningful look.This time the lesson went home.",
      "Question": "If _______ sticks come together it is impossible to break them said the old man giving his sons a meaningful look.This time the lesson went home."
    },
    {
      "Answer": "old man",
      "Original_Sentence": "If four sticks come together it is impossible to break them said the old man giving his sons a meaningful look.This time the lesson went home.",
      "Question": "If four sticks come together it is impossible to break them said the _______ giving his sons a meaningful look.This time the lesson went home."
    },
    {
      "Answer": "meaningful look.This",
      "Original_Sentence": "If four sticks come together it is impossible to break them said the old man giving his sons a meaningful look.This time the lesson went home.",
      "Question": "If four sticks come together it is impossible to break them said the old man giving his sons a _______ time the lesson went home."
    },
    {
      "Answer": "time",
      "Original_Sentence": "If four sticks come together it is impossible to break them said the old man giving his sons a meaningful look.This time the lesson went home.",
      "Question": "If four sticks come together it is impossible to break them said the old man giving his sons a meaningful look.This _______ the lesson went home."
    },
    {
      "Answer": "lesson",
      "Original_Sentence": "If four sticks come together it is impossible to break them said the old man giving his sons a meaningful look.This time the lesson went home.",
      "Question": "If four sticks come together it is impossible to break them said the old man giving his sons a meaningful look.This time the _______ went home."
    },
    {
      "Answer": "home",
      "Original_Sentence": "If four sticks come together it is impossible to break them said the old man giving his sons a meaningful look.This time the lesson went home.",
      "Question": "If four sticks come together it is impossible to break them said the old man giving his sons a meaningful look.This time the lesson went _______."
    }
  ]
}
```

## Contributers: 
* [Akash Verma](https://github.com/akashvermaofskt)
* Anurag Shakya
* [Mohit Rai](https://github.com/cenation092)
