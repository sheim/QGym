This is the readme file for the available URDFs of the Minicheetah. 

## Options to view the URDFs

### Option 1: Matlab
- Run the matlab script in `````$GPU_GYM_PATH/resources/robots/mini_cheetah/urdf/evaluateURDFMiniCheetah.m`````
- With this code, one can visualize the colision and/or the visual meshes. So you can easily edit the URDF and visualize it with which ever configuration you want. 

### Option 2: Online
- View the URDF online via the following link https://gkjohnson.github.io/urdf-loaders/javascript/example/bundle/index.html
- With this link, you can visualize the visual only, but you can play with the joint configurations. 

### Option 3: MuJoCo
- Load the URDF through Q2 and use `scripts/play.py` with the MuJoCo CPU
  backend to inspect its collision geometry and joint motion.
- Use a task-level smoke test after editing the asset to confirm that the same
  URDF loads through the supported backend contract.

--- 

## The URDFs

### The Simple URDF ```mini_cheetah_simple.urdf```
- This is a work in progress, but the idea is similar to the Humanoid, we should have a URDF version of the MiniCheetah with a simple collision mesh.

### Inertia corrections

The
[upstream MIT dynamics model](https://github.com/mit-biomimetics/Cheetah-Software/blob/master/common/include/Dynamics/MiniCheetah.h)
gives the base inertia as
`diag(0.011253, 0.036203, 0.042673) kg m^2`. The previous URDF value
`iyy=0.362030` was a decimal-place typo and violated the principal-moment
triangle inequality.

The upstream CAD thigh tensor also violates that inequality slightly after
rounding. Its principal moments were projected onto the nearest strictly
physical tensor in Frobenius norm while preserving its principal axes. The
same corrected tensors are used in the simple and rotor URDFs so physics
engines do not silently condition them in different ways.

The four `0.01 kg` foot links previously had zero inertia. They are modeled as
uniform spheres matching their `0.0202 m` collision geometry, with their center
of mass at the collision-sphere center.
