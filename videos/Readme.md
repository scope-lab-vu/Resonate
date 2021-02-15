## Videos


* **Nominal-scene.gif** - A Carla scene with weather (cloud=0.0, precipitation=0.0, deposits=0.0), where the AV is expected to operate. The system monitors (B-VAE Assurance Monitor, blur detector and the occlusion detector) outputs remain low throughout. The risk remains low through the scene as the weather is nominal and the AV has no faults during operation. 

* **Nominal-scene-with-camera-occlusion.gif** - A Carla scene with weather (cloud=0.0, precipitation=0.0, deposits=0.0), where the AV is expected to operate. The occlusion detectors identify when the cameras get occluded and send, while the other monitors remain low. The risk remains low initially but it increases as the AV encounters a camera related occlusion faults. 

* **adverse-scene-high-brightness.gif** - A Carla scene with weather (cloud=0.0, precipitation=0.0, deposits=0.0), and adverse brightness, where the AV is expected to operate. The B-VAE assurance monitor detects the increase in the brightness and its martingale value increases, but the blur detector and the occlusion detector outputs remain low. The risk remains low initially but it increases as the scene gets adverse with high brightness.


Same videos in mp4 format is available in [mp4 videos](https://github.com/Shreyasramakrishna90/Resonate-Dynamic-Risk/tree/main/videos/mp4%20videos)
