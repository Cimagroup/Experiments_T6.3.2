# Task6.3.2
Task 6.3.2 repository for REliable &amp; eXplAinable Swarm Intelligence for People with Reduced mObility project (REXASI-PRO) (GRANT AGREEMENT NO.101070028)

We use Python 3.10.12. Necessary dependencies are in requeriments.txt

We can see two folder of experiments depending on the Navground Scenario (Playground to experiment with navigation algorithms, https://idsia-robotics.github.io/navground/_build/html/index.html), one for a corridor scenario and another one for cross scenario. 

- Corridor scenario: half of the agents need to travel towards one end of a straight corridor, and the other half towards the other end. The two ends are wrapped together, i.e., agents exiting from on side are reintroduced on the other side. State estimation and collisions both conform to this lattice. The scenario tests opposing flows of agents. Some behavior let the agents spontaneously organize in lanes of opposing flow. 

- Cross scenario: In this scenario, there are 4 target waypoints located at (-side/2, 0), (side/2, 0), (0, -side/2), and (0, side/2). Half of the agents are tasked to pendle between the two vertically aligned waypoints, and half between the horizontally aligned waypoints. The scenario tests how agents cross in the middle, where the 4 opposing flows meets.
