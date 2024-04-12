


# install biogears somewhere there are write permissions
#
# https://github.com/BioGearsEngine/core/releases
#
# unsure how it will be installed on mac/linux but on windows:
#
# find the bin folder in the biogears installation, update string in globals.py
# to point to it

import globals

DIR_BIOGEARS_BIN = globals.DIR_BIOGEARS_BIN



import os
import subprocess
import time
import shutil

DIR_PATIENTS = os.path.join(DIR_BIOGEARS_BIN, "patients")
DIR_CUSTOM_SCENARIOS = os.path.join(DIR_BIOGEARS_BIN, "Scenarios", "CustomAsthma")
DIR_CSV_DATA = os.path.join(DIR_BIOGEARS_BIN, "csv-data")
EXE_SCENARIO = os.path.join(DIR_BIOGEARS_BIN, "bg-scenario.exe")


#clear old custom scenarios and csv-data
if os.path.exists(DIR_CUSTOM_SCENARIOS):
    shutil.rmtree(DIR_CUSTOM_SCENARIOS)
if os.path.exists(DIR_CSV_DATA):
    shutil.rmtree(DIR_CSV_DATA)

# ensure directories exist
os.makedirs(DIR_CSV_DATA, exist_ok=True)
os.makedirs(DIR_CUSTOM_SCENARIOS, exist_ok=True)


def make_asthma_scenario_xml(name, samples_per_second, pre_attack_seconds,
                             attack_severity_percent, attack_duration_seconds,
                             post_attack_seconds):
    if attack_severity_percent == 0:
        # no asthma present! Just get data for normal conditions
        actions = f"""<Actions>
            <Action xsi:type="AdvanceTimeData">
            <Time value="{pre_attack_seconds + attack_duration_seconds + post_attack_seconds}" unit="s"/>
            </Action>
            </Actions>"""
    else:
        actions =     f"""<Actions>
            <Action xsi:type="AdvanceTimeData">
            <Time value="{pre_attack_seconds}" unit="s"/>
            </Action>
            <Action xsi:type="AsthmaAttackData">
            <Severity value="{attack_severity_percent}"/>
            </Action>
            <Action xsi:type="AdvanceTimeData">
            <Time value="{attack_duration_seconds}" unit="s"/>
            </Action>
            <Action xsi:type="AsthmaAttackData">
            <Severity value="0.0"/>
            </Action>
            <Action xsi:type="AdvanceTimeData">
            <Time value="{post_attack_seconds}" unit="s"/>
            </Action>
            </Actions>"""

    return f"""<?xml version="1.0" encoding="UTF-8"?>
    <Scenario xmlns="uri:/mil/tatrc/physiology/datamodel" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" contentVersion="BioGears_6.3.0-beta" xsi:schemaLocation="">
        <Name>{name}</Name>
        <Description>Patient is afflicted with an asthma attack</Description>
        <InitialParameters><PatientFile>StandardMale.xml</PatientFile></InitialParameters>
        
        <DataRequests xmlns="uri:/mil/tatrc/physiology/datamodel" 
            xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" 
            SamplesPerSecond="{samples_per_second}">
            <DataRequest xsi:type="PhysiologyDataRequestData" Name="MeanArterialPressure" Unit="mmHg" Precision="1"/>
            <DataRequest xsi:type="PhysiologyDataRequestData" Name="HeartRate" Unit="" Precision="1"/>
            <DataRequest xsi:type="PhysiologyDataRequestData" Name="SystolicArterialPressure" Unit="mmHg" Precision="0"/>
            <DataRequest xsi:type="PhysiologyDataRequestData" Name="DiastolicArterialPressure" Unit="mmHg" Precision="1"/>
            <DataRequest xsi:type="PhysiologyDataRequestData" Name="CardiacOutput" Unit="L/min" Precision="2"/>
            <DataRequest xsi:type="PhysiologyDataRequestData" Name="CentralVenousPressure" Unit="mmHg" Precision="2"/>
            <DataRequest xsi:type="PhysiologyDataRequestData" Name="TidalVolume" Unit="mL" Precision="3"/>
            <DataRequest xsi:type="PhysiologyDataRequestData" Name="TotalLungVolume" Unit="L" Precision="2"/>
            <DataRequest xsi:type="PhysiologyDataRequestData" Name="RespirationRate" Unit="1/min" Precision="2"/>
            <DataRequest xsi:type="PhysiologyDataRequestData" Name="OxygenSaturation" Unit="unitless" Precision="3"/>
            <DataRequest xsi:type="PhysiologyDataRequestData" Name="CoreTemperature" Unit="degC" Precision="1"/>
        </DataRequests>
        
        {actions}
    </Scenario>
    """

completed_processes = 0

asthma_cases = {
    "none": 0.,
    "mild": .1,
    "moderate": .3,
    "severe": .7,
    "life-threatening":.9
}



# create the XML scenario files for our custom asthma attacks
for case_name, case_severity in asthma_cases.items():
    xml = make_asthma_scenario_xml(case_name, 20, 10, case_severity, 50, 10)
    xml_file_location = os.path.join(DIR_CUSTOM_SCENARIOS, case_name + ".xml")
    print(f"creating xml for case: {case_name} at {xml_file_location}")
    with open(xml_file_location, 'a') as xml_file:
        xml_file.write(xml + '\n')


patient_files = [f for f in os.listdir(DIR_PATIENTS) if f.endswith(".xml")]

# run our custom scenarios on each patient
# put data for each scenario into its own folder
for scenario_name, _ in asthma_cases.items():
    scenario_file =  os.path.join(DIR_CUSTOM_SCENARIOS, scenario_name + ".xml")

    # ensure directories exist
    dir_scenario_csv_data = os.path.join(DIR_CSV_DATA, scenario_name)
    os.makedirs(dir_scenario_csv_data, exist_ok=True)

    for patient_file in patient_files:
        patient_name = os.path.splitext(os.path.basename(patient_file))[0]
        csv_file = os.path.abspath(os.path.join(dir_scenario_csv_data, patient_name))
        print(f"executing case {patient_name}_{scenario_name}")
        command = [
            EXE_SCENARIO,
            "--patient", patient_file,
            "--results", csv_file,
            "--quiet",
            scenario_file
        ]
        print(f"Executed: {' '.join(command)}")

        start_time = time.time()
        subprocess.run(command, cwd=DIR_BIOGEARS_BIN)
        execution_time = time.time() - start_time

        print(f"Took {execution_time:.2f} seconds")
