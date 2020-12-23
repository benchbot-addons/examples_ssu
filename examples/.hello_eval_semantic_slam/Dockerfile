# Use a base image which provides as many of your dependencies as possible
FROM benchbot/submission:base

# Install any extra dependencies you may have
# NIL

# Add the files from your solution
ADD ./hello_eval_semantic_slam .
ADD ./eval_semantic_slam_agent.py .
ADD ./ground_truth_miniroom_1.json .

# Make the command used to run your solution the default command
CMD ./hello_eval_semantic_slam
