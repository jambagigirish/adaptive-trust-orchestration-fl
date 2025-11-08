# Adaptive Trust Orchestration for CPS using Federated Learning  
Repository for prototype of federated trust orchestration across multi-cloud/edge CPS environment.  
## Modules  
- /client : Edge client node code (Python)  
- /aggregator : Aggregator server (Java)  
- /orchestrator : Policy engine (Java)  
## Getting Started  
1. Configure Config.yaml.  
2. Deploy AggregatorServer in cloud (e.g., AWS + Azure).  
3. Launch EdgeClient instances (simulate many clients).  
4. Launch PolicyOrchestrator.  
5. Monitor logs: trust scores, policy actions, drift events.  
