#### 动态模型（agent 视角）

run `e27f92867845` · commit `f1d30c93ef` · task `866ea181155acd6b1a6bf294adc04548` · seeds {"data": 0, "split_test_and_calib": 0}

| 指标 | 值 |
|---|---|
| success_acc | 0.945 |
| success_brier | 0.035 |
| success_brier_calibrated | 0.035 |
| holder_changed_recall | 0.854 |
| holder_unchanged_kept | 0.995 |
| holder_unknown_wrongly_determined | 0.001 |
| holder_gone_recall | 0.030 |
| attr_changed_acc | 0.807 |
| discover_recall | 0.261 |
| obs_gain_mae | 0.310 |
| obs_gain_mae_train_mean_baseline | 0.756 |

#### 动态模型（env 视角）

run `9ba915748234` · commit `f1d30c93ef` · task `866ea181155acd6b1a6bf294adc04548` · seeds {"data": 0, "split_test_and_calib": 0}

| 指标 | 值 |
|---|---|
| success_acc | 0.987 |
| success_brier | 0.007 |
| success_brier_calibrated | 0.007 |
| holder_changed_recall | 0.999 |
| holder_unchanged_kept | 1.000 |
| holder_unknown_wrongly_determined | — |
| holder_gone_recall | — |
| attr_changed_acc | 0.906 |
| discover_recall | — |
| obs_gain_mae | — |
| obs_gain_mae_train_mean_baseline | — |

#### 策略评测（100 个留出世界）

run `7a586e0e3c3e` · commit `f1d30c93ef` · task `1e37f3c3b72a7f6c2e43390d80d41910` · seeds {"train": 0, "demo": "demo_seed('demo', 0, 0..300)", "bc_holdout": "demo_seed('bc_holdout', 0, ...)", "eval": [900000, 900100]}

| 策略 | 平均回报 [世界聚类 95%] | goal_rate | initial_goal_rate | new_goal_achievement | maintenance_success | search_miss_rate | search_no_evidence_rate | unprovoked_attack_rate | false_claim_rate | invalid_loop_rate |
|---|---|---|---|---|---|---|---|---|---|---|
| random | -1.152 [-1.3381, -0.9615] | 0.352 (87/247) | 0.353 (110/312) | 0.417 (83/199) | 0.232 (23/99) | 0.904 (199/220) | 0.759 (167/220) | 0.547 (134/245) | 0.000 (0/69) | 0.022 (162/7410) |
| wait_only | -0.743 [-0.8092, -0.6787] | 0.300 (74/247) | 0.353 (110/312) | 0.000 (0/199) | 0.970 (96/99) | — | — | — | — | 0.000 (0/7410) |
| scripted | -0.264 [-0.424, -0.1068] | 0.627 (155/247) | 0.353 (110/312) | 0.633 (126/199) | 0.646 (64/99) | 0.000 (0/4) | 1.000 (4/4) | 0.000 (0/214) | 0.000 (0/8) | 0.000 (0/7410) |
| bc | -0.272 [-0.4245, -0.125] | 0.619 (153/247) | 0.353 (110/312) | 0.623 (124/199) | 0.646 (64/99) | — | — | 0.073 (13/178) | 0.000 (0/2) | 0.000 (0/7410) |
| ppo | -0.272 [-0.4249, -0.1211] | 0.603 (149/247) | 0.353 (110/312) | 0.633 (126/199) | 0.596 (59/99) | — | — | 0.215 (54/251) | 0.000 (0/4) | 0.002 (13/7410) |
| ppo_test_time_no_predictions | -0.315 [-0.4676, -0.1699] | 0.599 (148/247) | 0.353 (110/312) | 0.613 (122/199) | 0.636 (63/99) | — | — | 0.213 (58/272) | 0.000 (0/10) | 0.006 (43/7410) |

| 配对比较（同一批世界） | 差 | 95% 区间 | 判定 |
|---|---|---|---|
| ppo_vs_wait_only:mean_return | 0.4715 | [0.364, 0.5818] | different |
| ppo_vs_wait_only:goal_rate | 0.3036 | [0.2196, 0.3889] | different |
| ppo_vs_scripted:mean_return | -0.0077 | [-0.0602, 0.0475] | inconclusive |
| ppo_vs_scripted:goal_rate | -0.0243 | [-0.0569, 0.0081] | inconclusive |
| ppo_vs_bc:mean_return | 0.0002 | [-0.0539, 0.0567] | inconclusive |
| ppo_vs_bc:goal_rate | -0.0162 | [-0.0486, 0.0163] | equivalent_within_margin |
| ppo_vs_ppo_test_time_no_predictions:mean_return | 0.0432 | [-0.0079, 0.0984] | inconclusive |
| ppo_vs_ppo_test_time_no_predictions:goal_rate | 0.004 | [-0.0317, 0.0377] | equivalent_within_margin |
| scripted_vs_wait_only:goal_rate | 0.3279 | [0.244, 0.4115] | different |

#### 策略评测（100 个留出世界）

run `0c3e741082da` · commit `f1d30c93ef` · task `1e37f3c3b72a7f6c2e43390d80d41910` · seeds {"train": 1, "demo": "demo_seed('demo', 1, 0..300)", "bc_holdout": "demo_seed('bc_holdout', 1, ...)", "eval": [900000, 900100]}

| 策略 | 平均回报 [世界聚类 95%] | goal_rate | initial_goal_rate | new_goal_achievement | maintenance_success | search_miss_rate | search_no_evidence_rate | unprovoked_attack_rate | false_claim_rate | invalid_loop_rate |
|---|---|---|---|---|---|---|---|---|---|---|
| random | -1.182 [-1.3826, -0.9783] | 0.360 (89/247) | 0.353 (110/312) | 0.422 (84/199) | 0.283 (28/99) | 0.888 (207/233) | 0.682 (159/233) | 0.528 (123/233) | 0.134 (11/82) | 0.019 (143/7410) |
| wait_only | -0.743 [-0.8092, -0.6787] | 0.300 (74/247) | 0.353 (110/312) | 0.000 (0/199) | 0.970 (96/99) | — | — | — | — | 0.000 (0/7410) |
| scripted | -0.264 [-0.424, -0.1068] | 0.627 (155/247) | 0.353 (110/312) | 0.633 (126/199) | 0.646 (64/99) | 0.000 (0/4) | 1.000 (4/4) | 0.000 (0/214) | 0.000 (0/8) | 0.000 (0/7410) |
| bc | -0.333 [-0.4818, -0.1827] | 0.591 (146/247) | 0.353 (110/312) | 0.588 (117/199) | 0.646 (64/99) | — | — | 0.105 (19/181) | — | 0.000 (0/7410) |
| ppo | -0.275 [-0.4285, -0.1217] | 0.619 (153/247) | 0.353 (110/312) | 0.638 (127/199) | 0.636 (63/99) | — | — | 0.230 (68/296) | 0.000 (0/23) | 0.036 (264/7410) |
| ppo_test_time_no_predictions | -0.339 [-0.4948, -0.1809] | 0.611 (151/247) | 0.353 (110/312) | 0.608 (121/199) | 0.657 (65/99) | — | — | 0.343 (80/233) | 0.000 (0/20) | 0.068 (504/7410) |

| 配对比较（同一批世界） | 差 | 95% 区间 | 判定 |
|---|---|---|---|
| ppo_vs_wait_only:mean_return | 0.4681 | [0.3559, 0.5808] | different |
| ppo_vs_wait_only:goal_rate | 0.3198 | [0.2362, 0.4055] | different |
| ppo_vs_scripted:mean_return | -0.0111 | [-0.0567, 0.0352] | inconclusive |
| ppo_vs_scripted:goal_rate | -0.0081 | [-0.0359, 0.0199] | equivalent_within_margin |
| ppo_vs_bc:mean_return | 0.0578 | [-0.0089, 0.1343] | inconclusive |
| ppo_vs_bc:goal_rate | 0.0283 | [0.0, 0.0586] | inconclusive |
| ppo_vs_ppo_test_time_no_predictions:mean_return | 0.0638 | [0.0202, 0.1099] | different |
| ppo_vs_ppo_test_time_no_predictions:goal_rate | 0.0081 | [-0.0169, 0.037] | equivalent_within_margin |
| scripted_vs_wait_only:goal_rate | 0.3279 | [0.244, 0.4115] | different |

#### 策略评测（100 个留出世界）

run `bf6203183081` · commit `f1d30c93ef` · task `1e37f3c3b72a7f6c2e43390d80d41910` · seeds {"train": 2, "demo": "demo_seed('demo', 2, 0..300)", "bc_holdout": "demo_seed('bc_holdout', 2, ...)", "eval": [900000, 900100]}

| 策略 | 平均回报 [世界聚类 95%] | goal_rate | initial_goal_rate | new_goal_achievement | maintenance_success | search_miss_rate | search_no_evidence_rate | unprovoked_attack_rate | false_claim_rate | invalid_loop_rate |
|---|---|---|---|---|---|---|---|---|---|---|
| random | -1.113 [-1.2855, -0.9355] | 0.328 (81/247) | 0.353 (110/312) | 0.392 (78/199) | 0.222 (22/99) | 0.856 (166/194) | 0.701 (136/194) | 0.538 (128/238) | 0.097 (7/72) | 0.024 (175/7410) |
| wait_only | -0.743 [-0.8092, -0.6787] | 0.300 (74/247) | 0.353 (110/312) | 0.000 (0/199) | 0.970 (96/99) | — | — | — | — | 0.000 (0/7410) |
| scripted | -0.264 [-0.424, -0.1068] | 0.627 (155/247) | 0.353 (110/312) | 0.633 (126/199) | 0.646 (64/99) | 0.000 (0/4) | 1.000 (4/4) | 0.000 (0/214) | 0.000 (0/8) | 0.000 (0/7410) |
| bc | -0.322 [-0.4669, -0.1733] | 0.595 (147/247) | 0.353 (110/312) | 0.578 (115/199) | 0.657 (65/99) | — | — | 0.054 (6/110) | — | 0.000 (0/7410) |
| ppo | -0.270 [-0.4101, -0.1246] | 0.583 (144/247) | 0.353 (110/312) | 0.643 (128/199) | 0.545 (54/99) | — | — | 0.212 (78/367) | 0.000 (0/202) | 0.004 (32/7410) |
| ppo_test_time_no_predictions | -0.319 [-0.4655, -0.1675] | 0.559 (138/247) | 0.353 (110/312) | 0.608 (121/199) | 0.515 (51/99) | — | — | 0.311 (137/440) | 0.000 (0/245) | 0.008 (62/7410) |

| 配对比较（同一批世界） | 差 | 95% 区间 | 判定 |
|---|---|---|---|
| ppo_vs_wait_only:mean_return | 0.4732 | [0.3821, 0.5682] | different |
| ppo_vs_wait_only:goal_rate | 0.2834 | [0.2088, 0.3597] | different |
| ppo_vs_scripted:mean_return | -0.0059 | [-0.0599, 0.0534] | inconclusive |
| ppo_vs_scripted:goal_rate | -0.0445 | [-0.0816, -0.008] | different |
| ppo_vs_bc:mean_return | 0.0518 | [-0.0093, 0.1143] | inconclusive |
| ppo_vs_bc:goal_rate | -0.0121 | [-0.0484, 0.0242] | equivalent_within_margin |
| ppo_vs_ppo_test_time_no_predictions:mean_return | 0.0491 | [-0.0073, 0.103] | inconclusive |
| ppo_vs_ppo_test_time_no_predictions:goal_rate | 0.0243 | [-0.008, 0.056] | inconclusive |
| scripted_vs_wait_only:goal_rate | 0.3279 | [0.244, 0.4115] | different |

#### 策略评测（100 个留出世界）

run `b1e784013252` · commit `f1d30c93ef` · task `1e37f3c3b72a7f6c2e43390d80d41910` · seeds {"train": 0, "demo": "demo_seed('demo', 0, 0..300)", "bc_holdout": "demo_seed('bc_holdout', 0, ...)", "eval": [900000, 900100]}

| 策略 | 平均回报 [世界聚类 95%] | goal_rate | initial_goal_rate | new_goal_achievement | maintenance_success | search_miss_rate | search_no_evidence_rate | unprovoked_attack_rate | false_claim_rate | invalid_loop_rate |
|---|---|---|---|---|---|---|---|---|---|---|
| random | -1.152 [-1.3381, -0.9615] | 0.352 (87/247) | 0.353 (110/312) | 0.417 (83/199) | 0.232 (23/99) | 0.904 (199/220) | 0.759 (167/220) | 0.547 (134/245) | 0.000 (0/69) | 0.022 (162/7410) |
| wait_only | -0.743 [-0.8092, -0.6787] | 0.300 (74/247) | 0.353 (110/312) | 0.000 (0/199) | 0.970 (96/99) | — | — | — | — | 0.000 (0/7410) |
| scripted | -0.264 [-0.424, -0.1068] | 0.627 (155/247) | 0.353 (110/312) | 0.633 (126/199) | 0.646 (64/99) | 0.000 (0/4) | 1.000 (4/4) | 0.000 (0/214) | 0.000 (0/8) | 0.000 (0/7410) |
| bc | -0.316 [-0.467, -0.1658] | 0.607 (150/247) | 0.353 (110/312) | 0.583 (116/199) | 0.667 (66/99) | — | — | 0.122 (19/156) | 0.000 (0/2) | 0.000 (0/7410) |
| ppo | -0.248 [-0.3883, -0.1113] | 0.587 (145/247) | 0.353 (110/312) | 0.603 (120/199) | 0.606 (60/99) | — | — | 0.182 (102/560) | 0.000 (0/1) | 0.001 (4/7410) |
| ppo_test_time_no_predictions | -0.292 [-0.4359, -0.1416] | 0.579 (143/247) | 0.353 (110/312) | 0.623 (124/199) | 0.556 (55/99) | — | — | 0.280 (165/589) | 0.000 (0/1) | 0.002 (13/7410) |

| 配对比较（同一批世界） | 差 | 95% 区间 | 判定 |
|---|---|---|---|
| ppo_vs_wait_only:mean_return | 0.4947 | [0.3996, 0.5932] | different |
| ppo_vs_wait_only:goal_rate | 0.2874 | [0.2107, 0.3621] | different |
| ppo_vs_scripted:mean_return | 0.0156 | [-0.0613, 0.0951] | inconclusive |
| ppo_vs_scripted:goal_rate | -0.0405 | [-0.0956, 0.0123] | inconclusive |
| ppo_vs_bc:mean_return | 0.0676 | [-0.0088, 0.1447] | inconclusive |
| ppo_vs_bc:goal_rate | -0.0202 | [-0.0732, 0.032] | inconclusive |
| ppo_vs_ppo_test_time_no_predictions:mean_return | 0.0433 | [-0.0103, 0.1015] | inconclusive |
| ppo_vs_ppo_test_time_no_predictions:goal_rate | 0.0081 | [-0.0369, 0.0494] | equivalent_within_margin |
| scripted_vs_wait_only:goal_rate | 0.3279 | [0.244, 0.4115] | different |

#### 策略评测（100 个留出世界）

run `ab770d204ec6` · commit `f1d30c93ef` · task `1e37f3c3b72a7f6c2e43390d80d41910` · seeds {"train": 0, "demo": "demo_seed('demo', 0, 0..300)", "bc_holdout": "demo_seed('bc_holdout', 0, ...)", "eval": [900000, 900100]}

| 策略 | 平均回报 [世界聚类 95%] | goal_rate | initial_goal_rate | new_goal_achievement | maintenance_success | search_miss_rate | search_no_evidence_rate | unprovoked_attack_rate | false_claim_rate | invalid_loop_rate |
|---|---|---|---|---|---|---|---|---|---|---|
| random | -1.152 [-1.3381, -0.9615] | 0.352 (87/247) | 0.353 (110/312) | 0.417 (83/199) | 0.232 (23/99) | 0.904 (199/220) | 0.759 (167/220) | 0.547 (134/245) | 0.000 (0/69) | 0.022 (162/7410) |
| wait_only | -0.743 [-0.8092, -0.6787] | 0.300 (74/247) | 0.353 (110/312) | 0.000 (0/199) | 0.970 (96/99) | — | — | — | — | 0.000 (0/7410) |
| scripted | -0.264 [-0.424, -0.1068] | 0.627 (155/247) | 0.353 (110/312) | 0.633 (126/199) | 0.646 (64/99) | 0.000 (0/4) | 1.000 (4/4) | 0.000 (0/214) | 0.000 (0/8) | 0.000 (0/7410) |
| bc | -0.337 [-0.4925, -0.1849] | 0.599 (148/247) | 0.353 (110/312) | 0.573 (114/199) | 0.657 (65/99) | — | — | 0.102 (18/177) | — | 0.000 (0/7410) |
| ppo | -0.281 [-0.4409, -0.1215] | 0.611 (151/247) | 0.353 (110/312) | 0.623 (124/199) | 0.606 (60/99) | — | — | 0.235 (63/268) | 0.000 (0/27) | 0.000 (2/7410) |

| 配对比较（同一批世界） | 差 | 95% 区间 | 判定 |
|---|---|---|---|
| ppo_vs_wait_only:mean_return | 0.4625 | [0.3488, 0.5704] | different |
| ppo_vs_wait_only:goal_rate | 0.3117 | [0.2241, 0.3992] | different |
| ppo_vs_scripted:mean_return | -0.0166 | [-0.1024, 0.0701] | inconclusive |
| ppo_vs_scripted:goal_rate | -0.0162 | [-0.06, 0.0286] | inconclusive |
| ppo_vs_bc:mean_return | 0.0567 | [-0.0206, 0.1361] | inconclusive |
| ppo_vs_bc:goal_rate | 0.0121 | [-0.0323, 0.0615] | inconclusive |
| scripted_vs_wait_only:goal_rate | 0.3279 | [0.244, 0.4115] | different |

#### 策略评测（100 个留出世界）

run `d3d22e8f9185` · commit `f1d30c93ef` · task `6f20527961ee2c1507fe2f3178404f58` · seeds {"train": 0, "demo": "demo_seed('demo', 0, 0..300)", "bc_holdout": "demo_seed('bc_holdout', 0, ...)", "eval": [900000, 900100]}

| 策略 | 平均回报 [世界聚类 95%] | goal_rate | initial_goal_rate | new_goal_achievement | maintenance_success | search_miss_rate | search_no_evidence_rate | unprovoked_attack_rate | false_claim_rate | invalid_loop_rate |
|---|---|---|---|---|---|---|---|---|---|---|
| random | -1.148 [-1.3302, -0.9598] | 0.279 (69/247) | 0.353 (110/312) | 0.327 (65/199) | 0.283 (28/99) | 0.901 (192/213) | 0.770 (164/213) | 0.549 (129/235) | 0.000 (0/42) | 0.021 (153/7410) |
| wait_only | -0.743 [-0.8092, -0.6787] | 0.300 (74/247) | 0.353 (110/312) | 0.000 (0/199) | 0.970 (96/99) | — | — | — | — | 0.000 (0/7410) |
| scripted | -0.260 [-0.4211, -0.1033] | 0.624 (154/247) | 0.353 (110/312) | 0.628 (125/199) | 0.646 (64/99) | 0.000 (0/4) | 1.000 (4/4) | 0.000 (0/193) | 0.000 (0/8) | 0.000 (0/7410) |
| bc | -0.307 [-0.4555, -0.1626] | 0.603 (149/247) | 0.353 (110/312) | 0.593 (118/199) | 0.646 (64/99) | — | — | 0.074 (15/203) | 0.000 (0/4) | 0.000 (0/7410) |
| ppo | -0.263 [-0.4189, -0.1121] | 0.636 (157/247) | 0.353 (110/312) | 0.643 (128/199) | 0.667 (66/99) | — | — | 0.111 (31/280) | 0.000 (0/40) | 0.012 (92/7410) |

| 配对比较（同一批世界） | 差 | 95% 区间 | 判定 |
|---|---|---|---|
| ppo_vs_wait_only:mean_return | 0.4801 | [0.3704, 0.5872] | different |
| ppo_vs_wait_only:goal_rate | 0.336 | [0.2531, 0.4223] | different |
| ppo_vs_scripted:mean_return | -0.0031 | [-0.0694, 0.0636] | inconclusive |
| ppo_vs_scripted:goal_rate | 0.0121 | [-0.0286, 0.0531] | inconclusive |
| ppo_vs_bc:mean_return | 0.0435 | [-0.0119, 0.1057] | inconclusive |
| ppo_vs_bc:goal_rate | 0.0324 | [-0.0041, 0.0714] | inconclusive |
| scripted_vs_wait_only:goal_rate | 0.3239 | [0.2439, 0.4033] | different |

#### 策略评测（100 个留出世界）

run `5014f18417cf` · commit `f1d30c93ef` · task `6f20527961ee2c1507fe2f3178404f58` · seeds {"train": 0, "demo": "demo_seed('demo', 0, 0..300)", "bc_holdout": "demo_seed('bc_holdout', 0, ...)", "eval": [900000, 900100]}

| 策略 | 平均回报 [世界聚类 95%] | goal_rate | initial_goal_rate | new_goal_achievement | maintenance_success | search_miss_rate | search_no_evidence_rate | unprovoked_attack_rate | false_claim_rate | invalid_loop_rate |
|---|---|---|---|---|---|---|---|---|---|---|
| random | -1.148 [-1.3302, -0.9598] | 0.279 (69/247) | 0.353 (110/312) | 0.327 (65/199) | 0.283 (28/99) | 0.901 (192/213) | 0.770 (164/213) | 0.549 (129/235) | 0.000 (0/42) | 0.021 (153/7410) |
| wait_only | -0.743 [-0.8092, -0.6787] | 0.300 (74/247) | 0.353 (110/312) | 0.000 (0/199) | 0.970 (96/99) | — | — | — | — | 0.000 (0/7410) |
| scripted | -0.260 [-0.4211, -0.1033] | 0.624 (154/247) | 0.353 (110/312) | 0.628 (125/199) | 0.646 (64/99) | 0.000 (0/4) | 1.000 (4/4) | 0.000 (0/193) | 0.000 (0/8) | 0.000 (0/7410) |
| bc | -0.322 [-0.4776, -0.1673] | 0.591 (146/247) | 0.353 (110/312) | 0.588 (117/199) | 0.636 (63/99) | — | — | 0.123 (26/212) | — | 0.000 (0/7410) |
| ppo | -0.261 [-0.4106, -0.1075] | 0.632 (156/247) | 0.353 (110/312) | 0.633 (126/199) | 0.626 (62/99) | — | — | 0.208 (84/404) | 0.000 (0/117) | 0.018 (133/7410) |
| ppo_test_time_no_predictions | -0.275 [-0.4236, -0.1274] | 0.619 (153/247) | 0.353 (110/312) | 0.638 (127/199) | 0.596 (59/99) | — | — | 0.177 (74/417) | 0.000 (0/141) | 0.024 (174/7410) |

| 配对比较（同一批世界） | 差 | 95% 区间 | 判定 |
|---|---|---|---|
| ppo_vs_wait_only:mean_return | 0.4822 | [0.3758, 0.5919] | different |
| ppo_vs_wait_only:goal_rate | 0.332 | [0.2451, 0.417] | different |
| ppo_vs_scripted:mean_return | -0.0011 | [-0.071, 0.0645] | inconclusive |
| ppo_vs_scripted:goal_rate | 0.0081 | [-0.0367, 0.0484] | equivalent_within_margin |
| ppo_vs_bc:mean_return | 0.0608 | [-0.0193, 0.1415] | inconclusive |
| ppo_vs_bc:goal_rate | 0.0405 | [0.0, 0.0788] | inconclusive |
| ppo_vs_ppo_test_time_no_predictions:mean_return | 0.0143 | [-0.0399, 0.076] | inconclusive |
| ppo_vs_ppo_test_time_no_predictions:goal_rate | 0.0121 | [-0.016, 0.041] | equivalent_within_margin |
| scripted_vs_wait_only:goal_rate | 0.3239 | [0.2439, 0.4033] | different |

#### 跨训练种子（均值 ± 标准差）

##### f1d30c93ef·1e37f3c3b72a7f6c2e43390d80d41910·cfg eff68cd0

runs `7a586e0e3c3e`, `0c3e741082da`, `bf6203183081` · train seeds [0, 1, 2]

| 策略 | mean_return | goal_rate | initial_goal_rate | new_goal_achievement | maintenance_success | search_miss_rate | search_no_evidence_rate | unprovoked_attack_rate | false_claim_rate | invalid_loop_rate |
|---|---|---|---|---|---|---|---|---|---|---|
| random | -1.149 ± 0.035 | 0.347 ± 0.017 | 0.353 ± 0.000 | 0.410 ± 0.016 | 0.246 ± 0.033 | 0.883 ± 0.025 | 0.714 ± 0.040 | 0.537 ± 0.009 | 0.077 ± 0.069 | 0.022 ± 0.002 |
| wait_only | -0.743 ± 0.000 | 0.300 ± 0.000 | 0.353 ± 0.000 | 0.000 ± 0.000 | 0.970 ± 0.000 | — | — | — | — | 0.000 ± 0.000 |
| scripted | -0.264 ± 0.000 | 0.627 ± 0.000 | 0.353 ± 0.000 | 0.633 ± 0.000 | 0.646 ± 0.000 | 0.000 ± 0.000 | 1.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 |
| bc | -0.309 ± 0.033 | 0.602 ± 0.015 | 0.353 ± 0.000 | 0.596 ± 0.024 | 0.650 ± 0.006 | — | — | 0.077 ± 0.025 | 0.000 (n=1) | 0.000 ± 0.000 |
| ppo | -0.272 ± 0.003 | 0.602 ± 0.018 | 0.353 ± 0.000 | 0.638 ± 0.005 | 0.593 ± 0.045 | — | — | 0.219 ± 0.009 | 0.000 ± 0.000 | 0.014 ± 0.019 |
| ppo_test_time_no_predictions | -0.324 ± 0.013 | 0.590 ± 0.028 | 0.353 ± 0.000 | 0.610 ± 0.003 | 0.603 ± 0.076 | — | — | 0.289 ± 0.068 | 0.000 ± 0.000 | 0.027 ± 0.035 |

#### 跨运行配对比较（同一批留出世界，A − B）

| A | B | 配置差异 | 同一提交 | 目标达成率差 [95%] 判定 | 平均回报差 [95%] 判定 |
|---|---|---|---|---|---|
| `7a586e0e3c3e` | `ab770d204ec6` | ablate_predictions | True | -0.0081 [-0.0514, 0.0329] inconclusive | 0.0089 [-0.0629, 0.0841] inconclusive |
| `7a586e0e3c3e` | `b1e784013252` | ablate_memory | True | 0.0162 [-0.0412, 0.0735] inconclusive | -0.0233 [-0.1053, 0.0593] inconclusive |
| `5014f18417cf` | `d3d22e8f9185` | ablate_predictions | True | -0.004 [-0.0533, 0.0415] inconclusive | 0.002 [-0.0828, 0.0846] inconclusive |
