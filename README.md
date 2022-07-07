# Text Games Synthetic Data Generation based on Behavior Profiles

Bachelor Thesis, July 2022

## Abstract

Data generation is a technique used to cope with the lack of real data and to improve the models trained on that data. However, most of the techniques for data generation on games either require pre-existing data, thus usually suffering from bias and low variability, or are focused on obtaining high scores, which is not suitable for games where the focus is on decision-making or where rewards are not applicable, such as serious games. We present an architecture to generate synthetic data suitable for decision-making games. Our model tackles the above-stated limitations by creating agents that imitate real players' behavior and gathering the data as they play the game. Therefore, our approach does not require pre-existing data. The agents are each given a behavior profile regarding several character traits they use for decision-making. The agents select the actions that best align with their profile's character traits while also considering past actions. In addition, this architecture incorporates behavior information into the generated data and, therefore, into the models trained with it. We illustrate this architecture's application by implementing it on a serious text game aimed at educating about cybercrimes, particularly *Online Grooming*. We generate a synthetic dataset for the game and train various Machine Learning models on it. The generated data successfully teaches these models to predict the behavior profile that generated each game run given as input the decisions taken. These models can then be used to find the degree of similarity of the real players' behavior with respect to the given profiles.

## References

`yarnScripts` folder contains the Online Grooming text game from ["Creación de un videojuego interactivo para educar en el uso responsable de las redes sociales" (Urbistondo Murua, Ana Leticia)](https://repositorio.comillas.edu/xmlui/handle/11531/41473).
