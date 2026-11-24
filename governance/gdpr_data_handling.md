# GDPR Data-Handling Note

**Scope.** ThermalGuard processes machine telemetry (NASA C-MAPSS, simulated and public domain) and
equipment documentation. The training data contains **no personal data**.

**Where personal data could appear in a real deployment**
- Maintenance logs: technician names, IDs and free-text notes.
- Assistant queries: operator usernames and question text.
- Logs: IP addresses in API Gateway and Lambda logs.

**Controls (by design)**
1. **Data minimisation:** the RUL models use sensor values and cycle counts only. No operator identifiers reach the feature tables.
2. **Pseudonymisation:** technician names in maintenance logs are replaced with role tokens (e.g. `TECH_07`) before indexing in the RAG vector store.
3. **Retention:** API logs are kept 14 days (CloudWatch retention set in Terraform). Assistant query logs are kept 30 days for evaluation, then deleted.
4. **Lawful basis:** legitimate interest (equipment safety and uptime). Operators are informed through the dashboard footer.
5. **Data-subject rights:** pseudonym mapping is held separately, so access and erasure requests can be fulfilled by deleting the mapping and the matching log entries.
6. **Processors and transfers:** the LLM API provider receives only question text plus retrieved manual chunks, never raw logs with identifiers. Use an EU region or provider DPA for production.
7. **Security:** secrets live in GitHub Actions secrets and AWS environment variables, never in the repo. IAM is least-privilege (Terraform).
