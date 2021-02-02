import nmpi
client = nmpi.Client("rafaperez")
job = client.submit_job(source="https://github.com/Matthijspals/Spin",
                        platform=nmpi.SPINNAKER,
                        #config = {"extra_pip_installs": ["snntoolbox"]},
                        collab_id=89273)
