import nmpi
client = nmpi.Client("MatthijsPals")
job = client.submit_job(source="https://github.com/Matthijspals/Spin",
                        platform=nmpi.SPINNAKER,
                        #config = {"extra_pip_installs": ["snntoolbox"]},
                        collab_id=89105,
                        command="run.py")
