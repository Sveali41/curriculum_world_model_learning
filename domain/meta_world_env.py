import metaworld
import time
import metaworld.policies

ml1 = metaworld.ML1('reach-v3')
env = ml1.train_classes['reach-v3'](render_mode='human')
task = ml1.train_tasks[0]
env.set_task(task)

policy = metaworld.policies.SawyerReachV3Policy()

obs, info = env.reset()

try:
    for i in range(3000):
        action = policy.get_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        time.sleep(0.05) 

        if terminated or truncated:
            print("Task finished, resetting environment...")
            obs, info = env.reset()

except KeyboardInterrupt:
    print("User manually stopped")

finally:
    # Close the window and release resources
    env.close()