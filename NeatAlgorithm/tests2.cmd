@REM FOR /L %%P IN (10,20,100) DO (FOR /L %%I IN (1,1,3) DO python main.py -e ConfigurableAnimalEnv -a NEAT -ak pop_size=%%P gen_max=50 selection_rate=0.3 conn_rate=0.5 node_rate=0.2 time_max=100 -s ./saves2/pop-size_neat_%%P_%%I)

FOR /L %%P IN (10,20,100) DO python main.py -e ConfigurableAnimalEnv -a NEAT -ak pop_size=%%P gen_max=1 selection_rate=0.3 conn_rate=0.5 node_rate=0.2 time_max=100 -s ./saves2/pop-size-time_neat_%%P

@REM FOR /L %%S IN (1,2,10) DO (FOR /L %%I IN (1,1,3) DO python main.py -e ConfigurableAnimalEnv -a NEAT -ak pop_size=50 gen_max=50 selection_rate=0.%%S conn_rate=0.5 node_rate=0.2 time_max=100 -s ./saves2/selection-rate_neat_0.%%S_%%I)

@REM FOR /L %%C IN (1,2,10) DO (FOR /L %%N IN (1,2,10) DO (FOR /L %%I IN (1,1,3) DO python main.py -e ConfigurableAnimalEnv -a NEAT -ak pop_size=50 gen_max=50 selection_rate=0.3 conn_rate=0.%%C node_rate=0.%%N time_max=100 -s ./saves2/conn-node_neat_0.%%C-0.%%N_%%I))
