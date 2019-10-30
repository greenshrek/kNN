# Environment
    - Use python3.7

# Files
- Part1.py : Basic KNN execution is implemented

- Part2.py : Distance Weighted KNN Classification is implemented. No implementation of normalized feature vectors.

- Part2_b.py : Distance Weighted KNN Classification is implemented. Implementation of normalized feature vectors.

- Part3.py : Distance Weighted KNN Regression is implemented. No implementation of normalized feature vectors.

- Part3_b.py : Distance Weighted KNN Regression is implemented. Implementation of normalized feature vectors.

- runner.py : main file from where the code can be executed in bulk. More description in the execution section.

- test_results.txt :  This file contains the results of the execution.

- respresentation.py : This file is used to draw graphical representation of the results data. test_results.txt serves as an input to this.

# Execution
- You can run the algorithm using runner.py It is by default set to below configuration. This default execution was used in many runs for this experiment. You can alter the value to do a quick and easy run.
    ```
    list_files = ['Part1.py', 'Part2.py', 'Part3.py']
    k = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    ```
- Simple combination for quick run.
    Copy the below configuration and paste ir in runner.py
    ```
    list_files = ['Part1.py']
    k = [1]
    ```
    Further, run the file using command below
     ```python runner.py```

- Once the execution of runner.py is complete you can see the results in test_results.txt file

- If you want to see the graphical representation please execute representaion.py using the command
```python representation.py``` You should enter the correct parameters to see the expected results.
