After chaining the input and output paths and selecting an implemented `dspy.signature` class (defines prompting input type), e.g `default_physical` and a `dspy.Prediction` (`dspy.Predict`, `dspy.ChainOfThough`, `dspy.ChainOfThoughWithHint`) we run the script either in local enviroment or in a [`Singularity`](https://github.com/UL-FRI-NLP-2023-2024/ul-fri-nlp-course-project-randomly_generated/blob/main/Singularity/Singularity-dspy.def) contiainer.

For more examples and instructions see the comments in `prompt.py`.
