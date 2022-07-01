# Contributions

Contributions are always welcome :smile: ! you can contribute in many ways. 

## Reporting Bugs

Report Bugs at https://github.com/sforaidl/jeta/issues

When reporting bugs please 
- Include the version and some information about your local environment setup. 
- Way to reproduce the bug

## Fixing Bugs: 

Issues tagged with `help wanted`, `good first issues`, and `bug` are open for everyone.

## Get Started

Follow the steps to setup jeta for local development process

1) Fork the Jeta repo on github

2) clone your fork on local machine using following terminal command. 
```bash
git clone https://github.com/<insert your github username>/jeta.git
```

3) Create a conda virtual environment using `environment.yml` file. 

```bash
conda env create -f environment.yml
conda activate jeta_env
cd jeta/
python setup.py develop
```

4) Setup pre-commit hooks

```bash
pip install pre-commit
pre-commit install
```

5) Create a branch to work on specific issue

```bash 
git checkout -b <branch name>
```

5) Add tests for the code you added and check the tests before commting. To run these tests, run following command from your root directory. 

```bash
pytest tests
```

6) Run pre-commit hooks to format your code into proper structure. 
```bash
pre-commit run --all-files
```

7) Commit your changes and push your branch to github
```bash
git add .
git commit -m "Description about the changes"
git push origin <branch name>
```

8) Open up a pull request on github. 
If all previous checks and tests pass, please send a pull request.


