# Imitation Learning Techniques

In this exercise you'll use imitation learning to teach a student policy to mimic an expert demonstrator. This is an important technique in robotics research.

We'll first try the behavioural cloning technique, which is a simple baseline for imitation learning. It can generate good policies, but they typically can't recover after making mistakes.

We'll then try the DAGGER algorithm, which results in policies that can recover from their mistakes!

You can then try the exercises at the end. 

### Instructions

- Install pybullet with `pip install pybullet`

- Open `Behavioural Cloning and Dagger.ipynb` to see an implementation of behavioural cloning and DAGGER with pybullet gym's Humanoid Flagrun environment.

You'll train a humanoid runner to mimic an expert demonstrator, producing an agent that can run towards targets, and get up after falling over.

**You can run this notebook on a laptop CPU -- no GPU needed!**

### Results

You'll end up with a trained student policy that can recover from its mistakes!

![](flagrun_adv_fallover.gif)
