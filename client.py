import nmpi
client = nmpi.Client("rafaperez")
job = client.submit_job(source="https://github.com/rafapb97/Spin",
                        platform=nmpi.SPINNAKER,
                        #config = {"extra_pip_installs": ["snntoolbox"]},
                        collab_id=89273,
                        command="runAcc.py")
