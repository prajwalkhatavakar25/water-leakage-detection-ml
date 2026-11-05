# Data Dictionary

| Feature | Data Type | Description |
|---|---|---|
| Time | Integer | Sequential timestamp index |
| Tank_ID | Integer | Identifier of one of the 100 tanks |
| Pressure | Float | Current tank pressure reading |
| Pressure_Change | Float | Difference between current and previous pressure |
| Refill | Binary | 1 indicates refill; 0 indicates no refill |
| Leak | Binary | 1 indicates leakage; 0 indicates no leakage |

## Target Variable

Leak is the binary classification target.

## Feature Engineering

Pressure Change is calculated as:

Pressure_Change(t) = Pressure(t) - Pressure(t-1)

The first available observation for each tank
uses zero as its initial pressure-change value.

## Model Input Features

Pressure, Pressure_Change and Refill.

Time and Tank_ID are retained for tracking
but excluded from the current model inputs.