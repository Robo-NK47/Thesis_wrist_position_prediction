# IMU based 3D space goal prediction system

This project is the second part of my thesis. 
The data originates from a system which I have designed, containing two IMU sensors,
one on the user's wrist and anothe on the user's elbow (both on the same arm).

The data than is being transfomed into a dataset, going thru various pre-processing operations along the way
(to the choosing of the user) and then fed into a script which trains an optimized MLP model (using optuna & pytorch).

The goal on the MLP model is to find the CURRENT location of the user's wrist.
Than, using that model, another dataset is made, this time in favor of an optimized LSTM model (again, using optuna & pytorch).
The LSTM model's goal is to predict the location of the user's wrist at the end of her/his motion.