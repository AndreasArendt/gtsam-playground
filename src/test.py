import gtsam
print("Imported successfully!")

# Try constructing a simple Pose2
p = gtsam.Pose2(1.0, 2.0, 0.5)
print("Pose2 works:", p)

# Try accessing a GTSAM class
graph = gtsam.NonlinearFactorGraph()
print("Graph OK:", graph)