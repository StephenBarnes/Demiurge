Demiurge
========================================================================

This is a Python program implementing a dialect of Scheme with a reflective tower, which I built for Stanford's CS242 (Programming Languages).

In "reflective tower", the "tower" is a lazily-generated infinite stack of interpreters interpreting interpreters, with the user's program at the very top.
The tower is "reflective" in the sense that any level can modify any lower level.

For example, a program `(+ 1 2)` will produce output 3; this is ordinary Scheme. Demiurge also allows us to write reflective programs such as `(joots (set eval_literal (lambda args "Malkovich")))`. The `joots` stands for "jump out of the system" (coined by Hofstadter) and allows us to modify the interpreter; this example overwrites the interpreter's `eval_literal` function so that all literals evaluate to the string "Malkovich". If we now run `(+ 1 2)` we will get the string "MalkovichMalkovich" as our output.

The system allows for arbitrary modifications of the interpreter, so we could for instance write a program which changes our interpreter into a Python interpreter, then write the rest of our program in Python. Or we could modify our interpreter's interpreter to interpret Python, then modify our interpreter to be a Bash interpreter written in Python, then put a Bash program on the top level. As far as I know, there are no practical applications of this, but it's fun.

For details on implementation, see the [final report](https://github.com/StephenBarnes/Demiurge/blob/master/final_report/StephenBarnes_CS242_Final_Project.pdf), or read the [source](https://github.com/StephenBarnes/Demiurge/blob/master/src/demiurge.py).
For examples of programs, see [src/examples](https://github.com/StephenBarnes/Demiurge/tree/master/src/examples).
For an overview of the system, the [presentation](https://github.com/StephenBarnes/Demiurge/blob/master/presentation/presentation.pdf) I gave might be helpful.
