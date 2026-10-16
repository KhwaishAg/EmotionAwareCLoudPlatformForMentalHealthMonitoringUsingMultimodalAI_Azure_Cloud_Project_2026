"""Optional free-tier Azure integrations used by the running app.

Every integration here is OFF unless its ``CLOUD_*`` environment variables are set
(on App Service they are app settings written by ``scripts/azure_free_services.py``).
With nothing configured the app behaves exactly as the zero-cost local version:
no module in this package opens a network connection.

* ``speech`` / ``language`` - Azure AI Speech (F0) transcription and Azure AI Language
  (F0) sentiment + key phrases, turned into a voice-modality contract by ``voice_content``.
* ``events`` - publishes an anonymised "assessment completed" event to an Event Grid
  topic; Event Grid delivers it back to the app's ``/events`` webhook.
* ``cosmos`` / ``history`` - stores those anonymised summaries in Cosmos DB (free tier)
  so a student can see their own history and trend, and delete it.

Only the Python standard library is used for HTTP, so no cloud SDK is installed.
"""
