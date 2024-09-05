# Fire Prediction

In places like Eswatini and Zimbabwe, there has been more damage caused by wild-fires so the automatic detection or prediction can be of drastic importance since that could mean many houses, livestock, and precious materials are saved. This challenges can help greatly towards the alleviation of this damage.

The two options for running this code uses the notebook but that can be done either locally or in a hosted environment like Google Colab but for data security concerns, I recommend you use the local environment. That could be determined by the computational resources available though, and it is worth mentioning that this code does not demand computationally expensive environments.

You are going to use the notebook called run.ipynb and make sure to select the kernel which has the resources or dependencies required for this code to run smoothly. For the dependencies, we have a file called requirements.txt which can be used to install them and the following command can do that for you.

```bash
python -m pip install -r requirements.txt
```

Once this has been completed, select the kernel which usese the python version where you installed these dependencies, then you need to execute all cells. The code is structured in a linear fashion therefore all the cells run from the first to the last one without giving you any trouble. During execution and when done, this code will generate a file which is a csv file. The name of the file will contain today's date, yes check the calander to confirm that! and it is this file that can be used for submission to the zindi platform to get the score and rank in the leaderboard.

As stated, when you opt for a hosted environment, you may need to change the file paths and upload the files which are Train.csv, Test.csv, and SampleSubmissoin.csv. These are mandatory for this code to execute properly and ensure that you have their paths configured right. [Let's just say you are the one going to change the file paths when using Google Colab - that's your penalty for risking data into proprietary service providers :)]