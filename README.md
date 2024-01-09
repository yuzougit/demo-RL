# Reinforcement Learning DDPG

Test of DDPG algorithm on a simplified satellite simulation.

---
## Project Structure

**train.py** -- Main loop between Environment and Agent   
**train_utils.py** -- Helper functions such as noise function       
**agent.py** -- Define how the agent makes actions given observations and updates policy         
**networks.py** -- Define actor and critic networks   
**prioritized_buffers** -- Implementation of PER   
**satellite_engine** -- Simplified satellite simulation   
**satellite_constant** -- Constants for calculating satellite simulation   
