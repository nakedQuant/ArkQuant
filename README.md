ArkQuant
由8个组件构成 gateway , pipe,finance, pb, simulation, risk, metrics, opt
gateway: assets , database , driver , spider
pipe : term, graph, pipeline, ump_picker , engine, pipeloader
finance: order, transaction, position, position_tracker, ledger, portfolio, account, slippage, commission, trading_controls, restrictions, execution
pb : underneath,  division, blotter, generator, broker
simulation : sim_params, clock, trading_simulation,
risk : alert , allocation, fuse,
metrics : metrics_set , metrics_tracker, tearsheet
opt : grid , similarity
main logic pipe : term --- compute point , pipeline constructed by term which joined by dependence , term composed by strat which expressed by indicators
out of pipeline is asset list 
risk : capital management ; risk alert , fuse
metrics : to measure portfolio return and ledger 
opt : to generate best performance args of pipeline and parallel to run trading algorithm on different args combination
