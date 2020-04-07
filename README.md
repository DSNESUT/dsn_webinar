# Web App For Monitoring Model Performance
App makes daily Scheduled queries to the table on database found at the db connection string.

Vulnerabilities are passed through data engineering pipeline and the then to the model to make prediction/classify. Model performance on queried data are dumped on the monitoring table.

App also contains a **Model** endpoint for live scoring/classification of issues.

## Files and variables to be created 
1. Enviromental variable "Connection"=" db connection string" (For the model app deployment)







