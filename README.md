## The Duet of Agents: Efficiently Boosting Mathematics Reasoning

![](image/image.jpg)

This is the repository of Group 60 for MIT 6.8610 final project.

### Configure the virtual environment

```bash
conda create -n Math_LLM python=3.11
conda activate Math_LLM
pip install -r requirements.txt
```

### Prepare your TogetherAI API KEY
Fill your keys to the list object `keys` under the file `API_call.py`.

### Run the experiment

#### Problem Decomposition
Run the following command:
```bash
python decomposition.py
```
The results will be shown under `./out/decomp_result`, which will be needed to run the solver.

#### MathReg
Run the following command:
```bash
python MathReg.py 
```

#### IntelliCode
Run the following command:
```bash
python IntelliCode.py 
```

#### Evaluate the result
After running the experiments, you can find the resulting files under `out` directory. Fill the paths into `scripts/evaluate.py` (we have filled some examples) and run the following command:
```bash
python scripts/evaluate.py
```
