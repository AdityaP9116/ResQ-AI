=== DIAG 3: Check if World classes differ ===
isaacsim.core.api.world.World = <class 'isaacsim.core.api.world.world.World'>
    id = 2861904163360
omni.isaac.core.world.World   = <class 'isaacsim.core.api.world.world.World'>
    id = 2861904163360
Same class? True
World_new._world_initialized = False
World_old._world_initialized = False
SimulationContext._instance = None

--- Creating World via PegasusInterface (uses World_new) ---
pg.world = <isaacsim.core.api.world.world.World object at 0x0000029A7FF3F850>
pg.world type = <class 'isaacsim.core.api.world.world.World'>
has _scene? True
World_new._world_initialized = True
World_old._world_initialized = True
SimulationContext._instance = <isaacsim.core.api.world.world.World object at 0x0000029A7FF3F850>
  id = 2862594914384
  pg.world is SimContext._instance? True
w.scene OK: <isaacsim.core.api.scenes.scene.Scene object at 0x0000029A6610AC50>

--- Now importing spawn_drone (module-level code will run) ---
After spawn_drone import:
World_new._world_initialized = True
World_old._world_initialized = True
SimulationContext._instance = <isaacsim.core.api.world.world.World object at 0x0000029A7FF3F850>
  id = 2862594914384
pg.world = <isaacsim.core.api.world.world.World object at 0x0000029A7FF3F850>
  pg.world is SimContext._instance? True
  has _scene? True
  has _scene (SimCtx instance)? True
pg.world.scene OK: <isaacsim.core.api.scenes.scene.Scene object at 0x0000029A6610AC50>

--- Checking PegasusInterface().world (as Vehicle.__init__ does) ---
PegasusInterface().world = <isaacsim.core.api.world.world.World object at 0x0000029A7FF3F850>
  id = 2862594914384
  has _scene? True
  .scene OK: <isaacsim.core.api.scenes.scene.Scene object at 0x0000029A6610AC50>