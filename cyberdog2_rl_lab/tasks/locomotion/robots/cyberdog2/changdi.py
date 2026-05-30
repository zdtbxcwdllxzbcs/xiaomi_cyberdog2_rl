 # =========================
    # 球（同样修复）
    # =========================
    ball = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/ball",

        # 👇 这里也要放！！
        collision_group=-1,

        spawn=sim_utils.SphereCfg(
            radius=0.15,

            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 0.2, 0.2),
            ),

            collision_props=sim_utils.CollisionPropertiesCfg(),

            rigid_props=sim_utils.RigidBodyPropertiesCfg(),

            mass_props=sim_utils.MassPropertiesCfg(
                mass=0.43
            ),

            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.5,
                dynamic_friction=1.2,
                restitution=0.6,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.3)
        ),
    )

   # =========================
    # 围栏（关键修复点在这里）
    # =========================

    wall_front = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/wall_front",

        # 👇 collision_group 放这里！！
        collision_group=-1,

        spawn=sim_utils.CuboidCfg(
            size=(FIELD_LENGTH, WALL_THICKNESS, WALL_HEIGHT),

            # 👇 这里只保留这个
            collision_props=sim_utils.CollisionPropertiesCfg(),

            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.2, 0.8, 0.2),
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.0, FIELD_WIDTH / 2, WALL_HEIGHT / 2)
        ),
    )

    wall_back = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/wall_back",
        collision_group=-1,
        spawn=sim_utils.CuboidCfg(
            size=(FIELD_LENGTH, WALL_THICKNESS, WALL_HEIGHT),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.2, 0.8, 0.2),
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.0, -FIELD_WIDTH / 2, WALL_HEIGHT / 2)
        ),
    )

    wall_left = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/wall_left",
        collision_group=-1,
        spawn=sim_utils.CuboidCfg(
            size=(WALL_THICKNESS, FIELD_WIDTH, WALL_HEIGHT),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.2, 0.8, 0.2),
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(-FIELD_LENGTH / 2, 0.0, WALL_HEIGHT / 2)
        ),
    )

    wall_right = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/wall_right",
        collision_group=-1,
        spawn=sim_utils.CuboidCfg(
            size=(WALL_THICKNESS, FIELD_WIDTH, WALL_HEIGHT),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.2, 0.8, 0.2),
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(FIELD_LENGTH / 2, 0.0, WALL_HEIGHT / 2)
        ),
    )
