# stereoVisualOdometry

Many systems rely entirely on GPS-based services to complete specific tasks. Those systems base their navigation protocols on the GPS they have, so it is extremely important for those services to be working all the time. However, when those services fail, it is necessary to have some way of still completing a task. 

Our solution uses computer vision to solve the problem when the GPS does not work. We are using datasets from two cameras, that basically allow us to get the depth of an image and have a 3D space where the actual movement can be tracked.

At the moment there are still some changes that need to be made to make sure that the matching between images is correct.
