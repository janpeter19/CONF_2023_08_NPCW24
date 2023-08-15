# Figure - Simulation of IEC
#          with functions added to facilitate explorative simulation work
#
# Author: Jan Peter Axelsson
#------------------------------------------------------------------------------------------------------------------
# 2021-09-24 - Created
# 2021-09-27 - Include also elution phase
# 2021-09-29 - Structured column system slightly differently and updated for that
# 2021-10-01 - Updated system_info() with FMU-explore version
# 2021-10-01 - Updated diagrams and used uv_detector signal 
# 2021-11-12 - After talking with Karl Johan Brink I changed volume to area*height and scale by area
# 2021-11-13 - After talking with Karl Johan I also introduce linear flow rate u i.e. u = F/area
# 2021-11-13 - Also changed time unit from s to min all over and according to Karl Johan
# 2021-11-25 - Introduced diagram Elution-vs-volume and Elution-vs-volume-combined
# 2021-11-27 - Modifed for F, u, V 
# 2021-12-02 - Update for use of FluidMixerV in BPL ver 2.0.9 - beta 
# 2021-12-03 - Added simple plot of column outlet vs time to bridge to OpenModelica demo
# 2021-12-03 - Extended FMU-explore 0.8.6 for function disp() and dictionary parLocation[]
# 2021-12-13 - Extended newplot() with diagrams inclukding concductivity instead of ion concentration
# 2021-12-13 - Change unit from min to hours and also affect process parameters
# 2021-12-14 - Changed start and point for diagrams related to elution to be set automatically based on parDict
# 2021-12-17 - Now adjusted diagrams Elution-pooling
# 2021-12-18 - Changed back to use unit mL for all volumes and what Karl Johan wanted
# 2021-12-18 - Correction of disp() - now FMU-explore ver 0.8.7
# 2021-12-22 - Change of how flows are controlled and their parameters
# 2022-04-25 - Update to FMU-explore 0.9.0.
# 2022-04-25 - Take away variable scaling and read it off from mode() when needed
# 2022-04-25 - Introduce a switch called scale_volume that is true for using volume for switch events alt time
# 2022-04-27 - Modified disp() and describe() to handle that scale_volume is boolean
# 2022-04-28 - Tidy up newplot()
# 2022-04-29 - Corrected newplot()  'Elution-pooling' and 'Pooling', also improved error text par() and init() 
# 2022-05-01 - Corrected newplot() 'Time' to 'time' and import when Jupyter widget framework is used
# 2022-05-07 - Changed time scale from hours to min
# 2022-05-21 - Added to newplot 'Elution-conductivity-combined-all'
# 2022-10-07 - Updated for FMU-explore 0.9.5 with disp() that do not include extra parameters with parLocation
# 2022-11-11 - Updating handling of V, V_m and LFR
# 2022-11-12 - Updating newplot() with plotType 'Elution-vs-CV
# 2022-11-14 - Introduced possibilty to give intial value of ion E in section 1 a value to illustrate need to wash
# 2022-12-01 - Added parameters for control_pooling based on UV-levels in combination with a time_window
# 2022-12-03 - Introduced diagram plotType='Elutions-vs-CV-pooling'
# 2022-12-09 - Adjusted to skip pooling2 for a while
# 2022-12-12 - FNU-explore 0.9.6b test with extension to par() using dictionary parCheck
# 2022-12-12 - Changed uv_high and uv_low to start_uv and stop_uv and also changed parLocation after FMU update
# 2022-12-16 - Updated describe() test for 'chromatogoraph' as well as for 'liquidphase' for clarity
# 2023-01-28 - Include E_in_sample an idea from Karl Johan Brink
# 2023-02-03 - Include a switch gradient that if true produce a salt gradient and if false stepwise increase
# 2023-02-06 - Change to ControlDesrptionBuffer and corresponding changes in parDict and parLocation etc
# 2023-02-06 - Included relevant list of parCheck
# 2023-02-06 - Updated to FMU-explore 0.9.6e including parCheck...
# 2023-02-06 - Play with the idea of parCalc - but dropped
# 2023-04-05 - Update FMU-explore 0.9.7
# 2023-04-17 - Modify for FMU-explore 0.9.7 for FMPy
# 2023-04-18 - Modify for FMU-explore 0.9.8 for FMPy - update model_get() for Boolean variables
# 2023-05-31 - Adjusted to from importlib.meetadata import version
# 2023-06-02 - Add logging of a few variables
#------------------------------------------------------------------------------------------------------------------

#------------------------------------------------------------------------------------------------------------------
#  Framework
#------------------------------------------------------------------------------------------------------------------

# Setup framework
import sys
import platform
import locale
import numpy as np 
import matplotlib.pyplot as plt 

from fmpy import simulate_fmu
from fmpy import read_model_description
import fmpy as fmpy

from itertools import cycle
from importlib.metadata import version 

# Set the environment - for Linux a JSON-file in the FMU is read
if platform.system() == 'Linux': locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')

#------------------------------------------------------------------------------------------------------------------
#  Setup application FMU
#------------------------------------------------------------------------------------------------------------------

# Provde the right FMU and load for different platforms in user dialogue:
global fmu_model
if platform.system() == 'Windows':
   print('Windows - run FMU pre-compiled JModelica 2.14')
   fmu_model ='BPL_IEC_Column_system_windows_jm_cs.fmu'       
   model_description = read_model_description(fmu_model)  
   flag_vendor = 'JM'
   flag_type = 'CS'
elif platform.system() == 'Linux':
#   flag_vendor = input('Linux - run FMU from JModelica (JM) or OpenModelica (OM)?')  
#   flag_type = input('Linux - run FMU-CS (CS) or ME (ME)?')  
#   print()   
   flag_vendor = 'OM'
   flag_type = 'ME'
   if flag_vendor in ['','JM','jm']:    
      print('Linux - run FMU pre-compiled JModelica 2.4')
      fmu_model ='BPL_IEC_Column_system_linux_jm_cs.fmu'      
      model_description = read_model_description(fmu_model) 
   if flag_vendor in ['OM','om']:
      print('Linux - run FMU pre-comiled OpenModelica 1.21.0') 
      if flag_type in ['CS','cs']:         
         fmu_model ='BPL_IEC_Column_system_linux_om_cs.fmu'    
         model_description = read_model_description(fmu_model) 
      if flag_type in ['ME','me']:         
         fmu_model ='BPL_IEC_Column_system_linux_om_me.fmu' 
         model_description = read_model_description(fmu_model) 
   else:    
      print('There is no FMU for this platform')

# Provide various opts-profiles
if flag_type in ['CS', 'cs']:
   opts_std = {'NCP': 500}
elif flag_type in ['ME', 'me']:
   opts_std = {'NCP': 500}
else:    
   print('There is no FMU for this platform')

# Provide various MSL and BPL versions
if flag_vendor in ['JM', 'jm']:
   constants = [v for v in model_description.modelVariables if v.causality == 'local'] 
   MSL_usage = [x[1] for x in [(constants[k].name, constants[k].start) \
                     for k in range(len(constants))] if 'MSL.usage' in x[0]][0]   
   MSL_version = [x[1] for x in [(constants[k].name, constants[k].start) \
                       for k in range(len(constants))] if 'MSL.version' in x[0]][0]
   BPL_version = [x[1] for x in [(constants[k].name, constants[k].start) \
                       for k in range(len(constants))] if 'BPL.version' in x[0]][0] 
elif flag_vendor in ['OM', 'om']:
   MSL_usage = '3.2.3 - used components: RealInput, RealOutput, CombiTimeTable, Types' 
   MSL_version = '3.2.3'
   BPL_version = 'Bioprocess Library version 2.1.1' 
else:    
   print('There is no FMU for this platform')

# Simulation time
global simulationTime; simulationTime = 100.0
global prevFinalTime; prevFinalTime = 0

# Dictionary of time discrete states
timeDiscreteStates = {} 

# Define a minimal compoent list of the model as a starting point for describe('parts')
component_list_minimum = []

#------------------------------------------------------------------------------------------------------------------
#  Specific application constructs: stateDict, parDict, diagrams, newplot(), describe()
#------------------------------------------------------------------------------------------------------------------
   
# Create stateDict that later will be used to store final state and used for initialization in 'cont':
global stateDict; stateDict =  {}
stateDict = {variable.derivative.name:None for variable in model_description.modelVariables \
                                            if variable.derivative is not None}
stateDict.update(timeDiscreteStates) 

global stateDictInitial; stateDictInitial = {}
for key in stateDict.keys():
    if not key[-1] == ']':
         if key[-3:] == 'I.y':
            stateDictInitial[key] = key[:-10]+'I_0'
         elif key[-3:] == 'D.x':
            stateDictInitial[key] = key[:-10]+'D_0'
         else:
            stateDictInitial[key] = key+'_0'
    elif key[-3] == '[':
        stateDictInitial[key] = key[:-3]+'_0'+key[-3:]
    elif key[-4] == '[':
        stateDictInitial[key] = key[:-4]+'_0'+key[-4:]
    elif key[-5] == '[':
        stateDictInitial[key] = key[:-5]+'_0'+key[-5:] 
    else:
        print('The state vector has more than 1000 states')
        break

global stateDictInitialLoc; stateDictInitialLoc = {}
for value in stateDictInitial.values():
    stateDictInitialLoc[value] = value

# Create dictionaries parDict and parLocation
global parDict; parDict = {}

parDict['diameter'] = 7.136
parDict['height'] = 20.0
parDict['x_m'] = 0.30
parDict['k1'] = 0.3
parDict['k2'] = 0.05
parDict['k3'] = 0.05
parDict['k4'] = 0.3
parDict['Q_av'] = 3.0

parDict['E_0'] = 0.0

parDict['P_in'] = 0.3
parDict['A_in'] = 0.3
parDict['E_in'] = 0
parDict['E_in_desorption_buffer'] = 0.3

parDict['LFR'] = 0.67

parDict['scale_volume'] = True
parDict['gradient'] = True
parDict['start_adsorption'] = 0
parDict['stop_adsorption'] = 67
parDict['start_desorption'] = 200
parDict['x_start_desorption'] = 0.2
parDict['stationary_desorption'] = 500
parDict['stop_desorption'] = 600
parDict['start_pooling'] = 308
parDict['stop_pooling'] = 600

#parDict['uv_start_trend'] = 0
parDict['start_uv'] = -1
parDict['stop_uv'] = -2

global parLocation; parLocation = {}
parLocation['diameter'] = 'column.diameter'
parLocation['height'] = 'column.height'
parLocation['x_m'] = 'column.x_m'
parLocation['k1'] = 'column.k1'
parLocation['k2'] = 'column.k2'
parLocation['k3'] = 'column.k3'
parLocation['k4'] = 'column.k4'
parLocation['Q_av'] = 'column.Q_av'

parLocation['E_0'] = 'column.column_section[1].c_0[3]'

parLocation['P_in'] = 'tank_sample.c_in[1]'
parLocation['A_in'] = 'tank_sample.c_in[2]'
parLocation['E_in'] = 'tank_sample.c_in[3]'
parLocation['E_in_desorption_buffer'] = 'tank_buffer2.c_in[3]'

parLocation['LFR'] = 'u'

parLocation['scale_volume'] = 'scale_volume'
parLocation['gradient'] = 'control_desorption_buffer.gradient'
parLocation['start_adsorption'] = 'control_sample.start'
parLocation['stop_adsorption'] = 'control_sample.stop'
parLocation['start_desorption'] = 'control_desorption_buffer.start'
parLocation['x_start_desorption'] = 'control_desorption_buffer.x_start'
parLocation['stationary_desorption'] = 'control_desorption_buffer.stationary'
parLocation['stop_desorption'] = 'control_desorption_buffer.stop'
parLocation['start_pooling'] = 'control_pooling.start'
parLocation['stop_pooling'] = 'control_pooling.stop'

#parLocation['uv_start_trend'] = 'control_pooling2.uv_start_trend'
parLocation['start_uv'] = 'control_pooling.start_uv_pooling'
parLocation['stop_uv'] = 'control_pooling.stop_uv_pooling'

# Extra only for describe()
global key_variables; key_variables = []
parLocation['V'] = 'column.V'; key_variables.append(parLocation['V'])
parLocation['scale_volume'] = 'scale_volume'; key_variables.append(parLocation['scale_volume'])
parLocation['VFR'] = 'F'; key_variables.append(parLocation['VFR'])
parLocation['area'] = 'column.area'; key_variables.append(parLocation['area'])
parLocation['V_m'] = 'column.V_m'; key_variables.append(parLocation['V_m'])

parLocation['column.column_section[1].V_m'] = 'column.column_section[1].V_m'; 
key_variables.append(parLocation['column.column_section[1].V_m'])

parLocation['tank_mixing.outlet.c[1]'] ='tank_mixing.outlet.c[1]'; 
key_variables.append(parLocation['tank_mixing.outlet.c[1]'])

parLocation['control_buffer2.scaling'] ='control_buffer2.scaling'; 
key_variables.append(parLocation['control_buffer2.scaling'])

# Parameter value check - especially for hysteresis to avoid runtime error
global parCheck; parCheck = []
parCheck.append("parDict['start_adsorption'] < parDict['stop_adsorption']")
parCheck.append("parDict['start_desorption'] < parDict['stationary_desorption']")
parCheck.append("parDict['stationary_desorption'] < parDict['stop_desorption']")
parCheck.append("parDict['start_uv'] > parDict['stop_uv']")


# Create list of diagrams to be plotted by simu()
global diagrams
diagrams = []

# Define standard plots
def profile(t_n, id):
    data = np.zeros(9)
    data[0] = sim_res['time'][t_n]
    for j in list(range(1,9)):
        data[j] = sim_res['column.column_section[' + str(j) + '].c[' + str(id) + ']'][t_n]
    return data

def newplot(title='IEC', plotType='Loading'):
   """ Standard plot window 
       title = '' """
   
   # Globals
   global ax1, ax2, ax3, ax4, ax5, ax6    
   global ax11, ax12, ax21, ax22
    
   # Reset pens
   setLines()

   # Plot diagram 
   if plotType == 'Loading':
         
      # Part of plot made before simulation
      plt.figure()
      ax1 = plt.subplot(2,1,1)
      ax2 = plt.subplot(2,1,2)
    
      ax1.set_title(title)
      ax1.grid()
      ax1.set_ylabel('c[PS] and c[AS][mg/mL]')
    
      ax2.grid()
      ax2.set_ylabel('c[PS] and c[AS][mg/mL]')
      ax2.set_xlabel('Sections in column - inlet to outlet') 
      
      # Part of plot made after simulation
      diagrams.clear()
      diagrams.append("ax1.plot(list(range(1,9)), profile(10,4)[1:], color='b', linestyle=linetype)")
      diagrams.append("ax1.plot(list(range(1,9)), profile(50,4)[1:], 'b')")
      diagrams.append("ax1.plot(list(range(1,9)), profile(150,4)[1:], 'b')")
      diagrams.append("ax1.plot(list(range(1,9)), profile(200,4)[1:], 'b')")
      diagrams.append("ax1.plot(list(range(1,9)), profile(250,4)[1:], 'b')")
      diagrams.append("ax1.plot(list(range(1,9)), profile(300,4)[1:], 'b')")
      diagrams.append("ax1.plot(list(range(1,9)), profile(350,4)[1:], 'b')")
      diagrams.append("ax1.plot(list(range(1,9)), profile(400,4)[1:], 'b')")
      diagrams.append("ax1.plot(list(range(1,9)), profile(450,4)[1:], 'b')")
      diagrams.append("ax1.plot(list(range(1,9)), profile(500,4)[1:], 'b')")
      diagrams.append("ax1.plot(list(range(1,9)), profile(10,5)[1:], 'r')")
      diagrams.append("ax1.plot(list(range(1,9)), profile(50,5)[1:], 'r')")
      diagrams.append("ax1.plot(list(range(1,9)), profile(150,5)[1:], 'r')")
      diagrams.append("ax1.plot(list(range(1,9)), profile(200,5)[1:], 'r')")
      diagrams.append("ax1.plot(list(range(1,9)), profile(250,5)[1:], 'r')")
      diagrams.append("ax1.plot(list(range(1,9)), profile(300,5)[1:], 'r')")
      diagrams.append("ax1.plot(list(range(1,9)), profile(350,5)[1:], 'r')")
      diagrams.append("ax1.plot(list(range(1,9)), profile(400,5)[1:], 'r')")
      diagrams.append("ax1.plot(list(range(1,9)), profile(450,5)[1:], 'r')")
      diagrams.append("ax1.plot(list(range(1,9)), profile(500,5)[1:], 'r')")
      diagrams.append("ax2.plot(list(range(1,9)), profile(500,4)[1:], 'b*-')")      
      diagrams.append("ax2.plot(list(range(1,9)), profile(500,5)[1:], 'r*-')")      
        
   elif plotType == 'Loading-combined':
      
      # Part of plot made before simulation   
      plt.figure()
      ax11 = plt.subplot(2,2,1)
      ax12 = plt.subplot(2,2,2)
      ax21 = plt.subplot(2,2,3)
      ax22 = plt.subplot(2,2,4)

      ax11.set_title(title)
      ax11.grid()
      ax11.set_ylabel('c[P] and c[A][mg/mL]')

      ax12.grid()
      ax12.set_ylabel('c[PS] and c[AS][mg/mL]')
           
      ax21.grid()
      ax21.set_ylabel('Tank_waste [mL]')
      ax21.set_xlabel('Time [min]')
   
      ax22.grid()
      ax22.set_ylabel('c[PS] and c[AS][mg/mL]')       
      ax22.set_xlabel('Section in column - inlet to outlet') 

      # Part of plot made after simulation
      diagrams.clear()    
      diagrams.append("ax11.plot(sim_res['time'], sim_res['tank_mixing.outlet.c[1]'], color='b', linestyle=linetype)")           
      diagrams.append("ax12.plot(list(range(1,9)), profile(10,4)[1:], color='b', linestyle=linetype)")
      diagrams.append("ax12.plot(list(range(1,9)), profile(50,4)[1:], color='b', linestyle=linetype)")
      diagrams.append("ax12.plot(list(range(1,9)), profile(150,4)[1:], color='b', linestyle=linetype)")
      diagrams.append("ax12.plot(list(range(1,9)), profile(200,4)[1:], color='b', linestyle=linetype)")
      diagrams.append("ax12.plot(list(range(1,9)), profile(250,4)[1:], color='b', linestyle=linetype)")
      diagrams.append("ax12.plot(list(range(1,9)), profile(300,4)[1:], color='b', linestyle=linetype)")
      diagrams.append("ax12.plot(list(range(1,9)), profile(350,4)[1:], color='b', linestyle=linetype)")
      diagrams.append("ax12.plot(list(range(1,9)), profile(400,4)[1:], color='b', linestyle=linetype)")
      diagrams.append("ax12.plot(list(range(1,9)), profile(450,4)[1:], color='b', linestyle=linetype)")
      diagrams.append("ax12.plot(list(range(1,9)), profile(500,4)[1:], color='b', linestyle=linetype)")
      diagrams.append("ax12.plot(list(range(1,9)), profile(10,5)[1:], color='r', linestyle=linetype)")
      diagrams.append("ax12.plot(list(range(1,9)), profile(50,5)[1:], color='r', linestyle=linetype)")
      diagrams.append("ax12.plot(list(range(1,9)), profile(150,5)[1:], color='r', linestyle=linetype)")
      diagrams.append("ax12.plot(list(range(1,9)), profile(200,5)[1:], color='r', linestyle=linetype)")
      diagrams.append("ax12.plot(list(range(1,9)), profile(250,5)[1:], color='r', linestyle=linetype)")
      diagrams.append("ax12.plot(list(range(1,9)), profile(300,5)[1:], color='r', linestyle=linetype)")
      diagrams.append("ax12.plot(list(range(1,9)), profile(350,5)[1:], color='r', linestyle=linetype)")
      diagrams.append("ax12.plot(list(range(1,9)), profile(400,5)[1:], color='r', linestyle=linetype)")
      diagrams.append("ax12.plot(list(range(1,9)), profile(450,5)[1:], color='r', linestyle=linetype)")
      diagrams.append("ax12.plot(list(range(1,9)), profile(500,5)[1:], color='r', linestyle=linetype)")
      diagrams.append("ax21.plot(sim_res['time'], sim_res['tank_waste.V'], color='b', linestyle=linetype)")
      diagrams.append("ax22.plot(list(range(1,9)), profile(500,4)[1:], color='b', linestyle=linetype)")      
      diagrams.append("ax22.plot(list(range(1,9)), profile(500,5)[1:], color='r', linestyle=linetype)")  
      
   elif plotType == 'Elution':
      
      # Part of plot made before simulation   
      plt.figure()
      ax1 = plt.subplot(2,1,1)
      ax2 = plt.subplot(2,1,2)
    
      ax1.set_title(title)
      ax1.grid()
      ax1.set_ylabel('c[P] and c[A]  [mg/mL]')
    
      ax2.grid()
      ax2.set_ylabel('c[P]+c[A] c[E]  [mg/mL]')
      ax2.set_xlabel('Time [min] - relative start desorption')       

      # Part of plot made after simulation
      diagrams.clear()    
      diagrams.append("ax1.plot(sim_res['time']-parDict['start_desorption']/model_get('control_buffer2.scaling'), \
                                sim_res['column.column_section[8].outlet.c[1]'], label='P', color='b', linestyle=linetype)")
      diagrams.append("ax1.plot(sim_res['time']-parDict['start_desorption']/model_get('control_buffer2.scaling'), \
                                sim_res['column.column_section[8].outlet.c[2]'], label='A', color='r', linestyle=linetype)")
      diagrams.append("ax1.set_xlim(left=0)")
      diagrams.append("ax1.set_ylim([0,0.45])")
      diagrams.append("ax1.legend()")
 
      diagrams.append("ax2.plot(sim_res['time']-parDict['start_desorption']/model_get('control_buffer2.scaling'), \
                                sim_res['uv_detector.value'], label='UV', color='k', linestyle=linetype)")
      diagrams.append("ax2.plot(sim_res['time']-parDict['start_desorption']/model_get('control_buffer2.scaling'), \
                           0.05*sim_res['column.column_section[8].outlet.c[3]'], label='salt', color='m', linestyle=linetype)")
      diagrams.append("ax2.set_xlim(left=0)") 
      diagrams.append("ax2.set_ylim([0,0.45])")
      diagrams.append("ax2.legend()")
      
   elif plotType == 'Elution-vs-volume':
         
      # Part of plot made before simulation   
      plt.figure()
      ax1 = plt.subplot(2,1,1)
      ax2 = plt.subplot(2,1,2)
    
      ax1.set_title(title)
      ax1.grid()
      ax1.set_ylabel('c[P] and c[A]  [mg/mL]')
    
      ax2.grid()
      ax2.set_ylabel('c[P]+c[A] c[E]  [mg/mL]')
      ax2.set_xlabel('Pumped liquid volume [mL] - relative start desorption')       

      # Part of plot made after simulation
      diagrams.clear()    
      diagrams.append("ax1.plot(sim_res['ackF'] - parDict['start_desorption']*model_get('F')/model_get('control_buffer2.scaling'), \
                                sim_res['column.column_section[8].outlet.c[1]'], label='P', color='b', linestyle=linetype)")
      diagrams.append("ax1.plot(sim_res['ackF'] - parDict['start_desorption']*model_get('F')/model_get('control_buffer2.scaling'), \
                                sim_res['column.column_section[8].outlet.c[2]'], label='A', color='r', linestyle=linetype)")
      diagrams.append("ax1.set_xlim(left=0)")
      diagrams.append("ax1.set_ylim([0,0.45])")
      diagrams.append("ax1.legend()")
 
      diagrams.append("ax2.plot(sim_res['ackF'] - parDict['start_desorption']*model_get('F')/model_get('control_buffer2.scaling'), \
                                sim_res['uv_detector.value'], label='UV', color='k', linestyle=linetype)")
      diagrams.append("ax2.plot(sim_res['ackF'] - parDict['start_desorption']*model_get('F')/model_get('control_buffer2.scaling'), \
                                0.05*sim_res['column.column_section[8].outlet.c[3]'], label='salt', color='m', linestyle=linetype)")
      diagrams.append("ax2.set_xlim(left=0)") 
      diagrams.append("ax2.set_ylim([0,0.45])")
      diagrams.append("ax2.legend()")

   elif plotType == 'Elution-vs-CV':
         
      # Part of plot made before simulation   
      plt.figure()
      ax1 = plt.subplot(2,1,1)
      ax2 = plt.subplot(2,1,2)
    
      ax1.set_title(title)
      ax1.grid()
      ax1.set_ylabel('c[P] and c[A]  [mg/mL]')
    
      ax2.grid()
      ax2.set_ylabel('c[P]+c[A] c[E]  [mg/mL]')
      ax2.set_xlabel('Pumped liquid volume [CV] - relative start desorption')       

      # Part of plot made after simulation
      diagrams.clear()    
      diagrams.append("ax1.plot((sim_res['ackF'] - parDict['start_desorption']*model_get('F')/model_get('control_buffer2.scaling'))/model_get('column.V'), \
                                sim_res['column.column_section[8].outlet.c[1]'], label='P', color='b', linestyle=linetype)")
      diagrams.append("ax1.plot((sim_res['ackF'] - parDict['start_desorption']*model_get('F')/model_get('control_buffer2.scaling'))/model_get('column.V'), \
                                sim_res['column.column_section[8].outlet.c[2]'], label='A', color='r', linestyle=linetype)")
      diagrams.append("ax1.set_xlim(left=0)")
      diagrams.append("ax1.set_ylim([0,0.45])")
      diagrams.append("ax1.legend()")
 
      diagrams.append("ax2.plot((sim_res['ackF'] - parDict['start_desorption']*model_get('F')/model_get('control_buffer2.scaling'))/model_get('column.V'), \
                                sim_res['uv_detector.value'], label='UV', color='k', linestyle=linetype)")
      diagrams.append("ax2.plot((sim_res['ackF'] - parDict['start_desorption']*model_get('F')/model_get('control_buffer2.scaling'))/model_get('column.V'), \
                                0.05*sim_res['column.column_section[8].outlet.c[3]'], label='salt', color='m', linestyle=linetype)")
      diagrams.append("ax2.set_xlim(left=0)") 
      diagrams.append("ax2.set_ylim([0,0.45])")
      diagrams.append("ax2.legend()")

   elif plotType == 'Elution-vs-volume-all':
         
      # Part of plot made before simulation   
      plt.figure()
      ax1 = plt.subplot(2,1,1)
      ax2 = plt.subplot(2,1,2)
    
      ax1.set_title(title)
      ax1.grid()
      ax1.set_ylabel('c[P] and c[A]  [mg/mL]')
    
      ax2.grid()
      ax2.set_ylabel('c[P]+c[A] c[E]  [mg/mL]')
      ax2.set_xlabel('Pumped liquid volume [mL]')       

      # Part of plot made after simulation
      diagrams.clear()    
      diagrams.append("ax1.plot(sim_res['ackF'], \
                                sim_res['column.column_section[8].outlet.c[1]'], label='P', color='b', linestyle=linetype)")
      diagrams.append("ax1.plot(sim_res['ackF'], \
                                sim_res['column.column_section[8].outlet.c[2]'], label='A', color='r', linestyle=linetype)")
      diagrams.append("ax1.set_xlim(left=0)")
      diagrams.append("ax1.set_ylim([0,0.45])")
      diagrams.append("ax1.legend()")
 
      diagrams.append("ax2.plot(sim_res['ackF'], \
                                sim_res['uv_detector.value'], label='UV', color='k', linestyle=linetype)")
      diagrams.append("ax2.plot(sim_res['ackF'], \
                                0.05*sim_res['column.column_section[8].outlet.c[3]'], label='salt', color='m', linestyle=linetype)")
      diagrams.append("ax2.set_xlim(left=0)") 
      diagrams.append("ax2.set_ylim([0,0.45])")
      diagrams.append("ax2.legend()")


   elif plotType == 'Elution-conductivity-vs-volume':
         
      # Part of plot made before simulation
      plt.figure()
      ax1 = plt.subplot(3,1,1)
      ax2 = plt.subplot(3,1,2)
      ax3 = plt.subplot(3,1,3)
    
      ax1.set_title(title)
      ax1.grid()
      ax1.set_ylabel('c[P] and c[A]  [mg/mL]')
    
      ax2.grid()
      ax2.set_ylabel('UV-detector []')
 
      ax3.grid()
      ax3.set_ylabel('Conductivity [mS/cm]')      
      ax3.set_xlabel('Pumped liquid volume [mL]')       

      # Part of plot made after simulation
      diagrams.clear()    
      diagrams.append("ax1.plot(sim_res['ackF'] - parDict['start_desorption']*model_get('F')/model_get('control_buffer2.scaling'), \
                                sim_res['column.column_section[8].outlet.c[1]'], label='P', color='b', linestyle=linetype)")
      diagrams.append("ax1.plot(sim_res['ackF'] - parDict['start_desorption']*model_get('F')/model_get('control_buffer2.scaling'), \
                                sim_res['column.column_section[8].outlet.c[2]'], label='A', color='r', linestyle=linetype)")
      diagrams.append("ax1.set_xlim(left=0)")
      diagrams.append("ax1.set_ylim([0,0.45])")
      diagrams.append("ax1.legend()")
 
      diagrams.append("ax2.plot(sim_res['ackF'] - parDict['start_desorption']*model_get('F')/model_get('control_buffer2.scaling'), \
                                sim_res['uv_detector.value'], label='UV', color='k', linestyle=linetype)")
      diagrams.append("ax2.set_xlim(left=0)") 
      diagrams.append("ax2.set_ylim([0,0.45])")

      diagrams.append("ax3.plot(sim_res['ackF'] - parDict['start_desorption']*model_get('F')/model_get('control_buffer2.scaling'), \
                                sim_res['conductivity_detector.value'], color='m', linestyle=linetype)")
      diagrams.append("ax3.set_xlim(left=0)") 

   elif plotType == 'Elution-conductivity-vs-volume-all':
         
      # Part of plot made before simulation
      plt.figure()
      ax1 = plt.subplot(3,1,1)
      ax2 = plt.subplot(3,1,2)
      ax3 = plt.subplot(3,1,3)
    
      ax1.set_title(title)
      ax1.grid()
      ax1.set_ylabel('c[P] and c[A]  [mg/mL]')
    
      ax2.grid()
      ax2.set_ylabel('UV-detector []')
 
      ax3.grid()
      ax3.set_ylabel('Conductivity [mS/cm]')      
      ax3.set_xlabel('Pumped liquid volume [mL]')       

      # Part of plot made after simulation
      diagrams.clear()    
      diagrams.append("ax1.plot(sim_res['ackF'], \
                                sim_res['column.column_section[8].outlet.c[1]'], label='P', color='b', linestyle=linetype)")
      diagrams.append("ax1.plot(sim_res['ackF'], \
                                sim_res['column.column_section[8].outlet.c[2]'], label='A', color='r', linestyle=linetype)")
      diagrams.append("ax1.set_xlim(left=0)")
      diagrams.append("ax1.legend()")
 
      diagrams.append("ax2.plot(sim_res['ackF'], \
                                sim_res['uv_detector.value'], label='UV', color='k', linestyle=linetype)")
      diagrams.append("ax2.set_xlim(left=0)") 

      diagrams.append("ax3.plot(sim_res['ackF'], \
                                sim_res['conductivity_detector.value'], color='m', linestyle=linetype)")
      diagrams.append("ax3.set_xlim(left=0)") 

   elif plotType == 'Elution-combined':
         
      # Part of plot made before simulation
      plt.figure()
      ax1 = plt.subplot(2,1,1)
      ax2 = plt.subplot(8,1,5)
      ax3 = plt.subplot(8,1,6)
      ax4 = plt.subplot(8,1,7)
      ax5 = plt.subplot(8,1,8)
    
      ax1.set_title(title)
      ax1.grid()
      ax1.set_ylabel('c[P], c[A], c[E] [mg/mL]')
    
      ax2.grid()
      ax2.set_ylabel('F sample [mL/min]')

      ax3.grid()
      ax3.set_ylabel('F buff1 [mL/min]')

      ax4.grid()
      ax4.set_ylabel('F buff2 [mL/min]')

      ax5.grid()
      ax5.set_ylabel('V prod [L]')
      ax5.set_xlabel('Time [min]')  

      # Part of plot made after simulation
      diagrams.clear()    
      diagrams.append("ax1.plot(sim_res['time'], sim_res['column.column_section[8].outlet.c[1]'], label='P', color='b', linestyle=linetype)")
      diagrams.append("ax1.plot(sim_res['time'], sim_res['column.column_section[8].outlet.c[2]'], label='A', color='r', linestyle=linetype)")
      diagrams.append("ax1.plot(sim_res['time'], 0.05*sim_res['column.column_section[8].outlet.c[3]'], label='E', color='m', linestyle=linetype)")
      diagrams.append("ax1.legend()")
      
      diagrams.append("ax2.step(sim_res['time'], sim_res['tank_sample.Fsp'], color='g', linestyle=linetype)")     
      diagrams.append("ax3.plot(sim_res['time'], sim_res['tank_buffer1.Fsp'], color='g', linestyle=linetype)")                
      diagrams.append("ax4.plot(sim_res['time'], sim_res['tank_buffer2.Fsp'], color='g', linestyle=linetype)") 
      diagrams.append("ax5.step(sim_res['time'], sim_res['tank_harvest.V'], color='g', linestyle=linetype)") 
  
   elif plotType == 'Elution-vs-volume-combined':
         
      # Part of plot made before simulation
      plt.figure()
      ax1 = plt.subplot(2,1,1)
      ax2 = plt.subplot(8,1,5)
      ax3 = plt.subplot(8,1,6)
      ax4 = plt.subplot(8,1,7)
      ax5 = plt.subplot(8,1,8)
    
      ax1.set_title(title)
      ax1.grid()
      ax1.set_ylabel('c[P], c[A], c[E] [mg/mL]')
    
      ax2.grid()
      ax2.set_ylabel('F sample')

      ax3.grid()
      ax3.set_ylabel('F buffer 1')

      ax4.grid()
      ax4.set_ylabel('F buffer 2')

      ax5.grid()
      ax5.set_ylabel('V harvest [mL]')
      ax5.set_xlabel('Pumped liquid volume [mL]')

      # Part of plot made after simulation
      diagrams.clear()    
      diagrams.append("ax1.plot(sim_res['ackF'] - parDict['start_desorption']*model_get('F')/model_get('control_buffer2.scaling'), \
                                sim_res['column.column_section[8].outlet.c[1]'], label='P', color='b', linestyle=linetype)")
      diagrams.append("ax1.plot(sim_res['ackF'] - parDict['start_desorption']*model_get('F')/model_get('control_buffer2.scaling'), \
                                sim_res['column.column_section[8].outlet.c[2]'], label='A', color='r', linestyle=linetype)")
      diagrams.append("ax1.plot(sim_res['ackF'] - parDict['start_desorption']*model_get('F')/model_get('control_buffer2.scaling'), \
                           0.05*sim_res['column.column_section[8].outlet.c[3]'], label='E', color='m', linestyle=linetype)")
      diagrams.append("ax1.legend()")
      
      diagrams.append("ax2.plot(sim_res['ackF'] - parDict['start_desorption']*model_get('F')/model_get('control_buffer2.scaling'), \
                                sim_res['tank_sample.Fsp'], color='g', linestyle=linetype)")     
      diagrams.append("ax3.plot(sim_res['ackF'] - parDict['start_desorption']*model_get('F')/model_get('control_buffer2.scaling'), \
                                sim_res['tank_buffer1.Fsp'], color='g', linestyle=linetype)")                
      diagrams.append("ax4.plot(sim_res['ackF'] - parDict['start_desorption']*model_get('F')/model_get('control_buffer2.scaling'), \
                                sim_res['tank_buffer2.Fsp'], color='g', linestyle=linetype)") 
      diagrams.append("ax5.plot(sim_res['ackF'] - parDict['start_desorption']*model_get('F')/model_get('control_buffer2.scaling'), \
                                sim_res['tank_harvest.V'], color='g', linestyle=linetype)") 

   elif plotType == 'Elution-conductivity-vs-volume-combined':
         
      # Part of plot made before simulation
      plt.figure()
      ax1 = plt.subplot(2,1,1)
      ax2 = plt.subplot(10,1,6)
      ax3 = plt.subplot(10,1,7)
      ax4 = plt.subplot(10,1,8)
      ax5 = plt.subplot(10,1,9)
      ax6 = plt.subplot(10,1,10)
 
      #ax1.set_title(title)
      ax1.grid()
      ax1.set_ylabel('c[P], c[A] [mg/mL]')

      ax2.grid()
      ax2.set_ylabel('c [mS/cm]')      

      ax3.grid()
      ax3.set_ylabel('F load [mL/min]')

      ax4.grid()
      ax4.set_ylabel('Fb1 [mL/min]')

      ax5.grid()
      ax5.set_ylabel('Fb2 [mL/min]')

      ax6.grid()
      ax6.set_ylabel('V [mL]')
      ax6.set_xlabel('Pumped liquid volume [mL]')

      # Part of plot made after simulation
      diagrams.clear()    
      diagrams.append("ax1.plot(sim_res['ackF'] - parDict['start_desorption']*model_get('F')/model_get('control_buffer2.scaling'), \
                                sim_res['column.column_section[8].outlet.c[1]'], label='P', color='b', linestyle=linetype)")
      diagrams.append("ax1.plot(sim_res['ackF'] - parDict['start_desorption']*model_get('F')/model_get('control_buffer2.scaling'), \
                                sim_res['column.column_section[8].outlet.c[2]'], label='A', color='r', linestyle=linetype)")
      diagrams.append("ax1.legend()")
      diagrams.append("ax1.set_ylim([0, 1.05*max(sim_res['column.column_section[8].outlet.c[1]'])])")
      
      diagrams.append("ax2.plot(sim_res['ackF'] - parDict['start_desorption']*model_get('F')/model_get('control_buffer2.scaling'), \
                                sim_res['conductivity_detector.value'], color='m', linestyle=linetype)")      
      diagrams.append("ax3.plot(sim_res['ackF'] - parDict['start_desorption']*model_get('F')/model_get('control_buffer2.scaling'), \
                                sim_res['tank_sample.Fsp'], color='g', linestyle=linetype)")     
      diagrams.append("ax4.plot(sim_res['ackF'] - parDict['start_desorption']*model_get('F')/model_get('control_buffer2.scaling'), \
                                sim_res['tank_buffer1.Fsp'], color='g', linestyle=linetype)")                
      diagrams.append("ax5.plot(sim_res['ackF'] - parDict['start_desorption']*model_get('F')/model_get('control_buffer2.scaling'), \
                                sim_res['tank_buffer2.Fsp'], color='g', linestyle=linetype)") 
      diagrams.append("ax6.plot(sim_res['ackF'] - parDict['start_desorption']*model_get('F')/model_get('control_buffer2.scaling'), \
                                sim_res['tank_harvest.V'], color='g', linestyle=linetype)") 
      diagrams.append("ax1.set_xlim(0)")
      diagrams.append("ax2.set_xlim(0)")
      diagrams.append("ax3.set_xlim(0)")
      diagrams.append("ax4.set_xlim(0)")
      diagrams.append("ax5.set_xlim(0)")
      diagrams.append("ax6.set_xlim(0)")

   elif plotType == 'Elution-conductivity-vs-volume-combined-all':
         
      # Part of plot made before simulation
      plt.figure()
      ax1 = plt.subplot(2,1,1)
      ax2 = plt.subplot(10,1,6)
      ax3 = plt.subplot(10,1,7)
      ax4 = plt.subplot(10,1,8)
      ax5 = plt.subplot(10,1,9)
      ax6 = plt.subplot(10,1,10)
 
      ax1.set_title(title)
      ax1.grid()
      ax1.set_ylabel('c[P], c[A] [mg/mL]')

      ax2.grid()
      ax2.set_ylabel('c [mS/cm]')      

      ax3.grid()
      ax3.set_ylabel('F load [mL/min]')

      ax4.grid()
      ax4.set_ylabel('Fb1 [mL/min]')

      ax5.grid()
      ax5.set_ylabel('Fb2 [mL/min]')

      ax6.grid()
      ax6.set_ylabel('V [mL]')
      ax6.set_xlabel('Pumped liquid volume [mL]')

      # Part of plot made after simulation
      diagrams.clear()    
      diagrams.append("ax1.plot(sim_res['ackF'], sim_res['column.column_section[8].outlet.c[1]'], label='P', color='b', linestyle=linetype)")
      diagrams.append("ax1.plot(sim_res['ackF'], sim_res['column.column_section[8].outlet.c[2]'], label='A', color='r', linestyle=linetype)")
      diagrams.append("ax1.legend()")
      diagrams.append("ax2.plot(sim_res['ackF'], sim_res['conductivity_detector.value'], color='m', linestyle=linetype)")      
      diagrams.append("ax3.step(sim_res['ackF'], sim_res['tank_sample.Fsp'], color='g', linestyle=linetype)")     
      diagrams.append("ax4.plot(sim_res['ackF'], sim_res['tank_buffer1.Fsp'], color='g', linestyle=linetype)")                
      diagrams.append("ax5.plot(sim_res['ackF'], sim_res['tank_buffer2.Fsp'], color='g', linestyle=linetype)") 
      diagrams.append("ax6.plot(sim_res['ackF'], sim_res['tank_harvest.V'], color='g', linestyle=linetype)") 

   elif plotType == 'Elution-conductivity-vs-CV-combined-all':
         
      # Part of plot made before simulation
      plt.figure()
      ax1 = plt.subplot(2,1,1)
      
      ax2 = plt.subplot(10,1,6)
      ax3 = plt.subplot(10,1,7)
      ax4 = plt.subplot(10,1,8)
      ax5 = plt.subplot(10,1,9)
      ax6 = plt.subplot(10,1,10)
      
      #ax2 = plt.subplot(8,1,4)
      #ax3 = plt.subplot(8,1,5)
      #ax4 = plt.subplot(8,1,6)
      #ax5 = plt.subplot(8,1,7)
      #ax6 = plt.subplot(8,1,8)
 
      ax1.set_title(title)
      ax1.grid()
      ax1.set_ylabel('c[P], c[A] [mg/mL]')

      ax2.grid()
      ax2.set_ylabel('c [mS/cm]')      

      ax3.grid()
      ax3.set_ylabel('F load [mL/min]')

      ax4.grid()
      ax4.set_ylabel('Fb1 [mL/min]')

      ax5.grid()
      ax5.set_ylabel('Fb2 [mL/min]')

      ax6.grid()
      ax6.set_ylabel('V [mL]')
      ax6.set_xlabel('Pumped liquid volume [CV]')

      # Part of plot made after simulation
      diagrams.clear()    
      diagrams.append("ax1.plot(sim_res['ackF']/model_get('column.V'), sim_res['column.column_section[8].outlet.c[1]'], label='P', color='b', linestyle=linetype)")
      diagrams.append("ax1.plot(sim_res['ackF']/model_get('column.V'), sim_res['column.column_section[8].outlet.c[2]'], label='A', color='r', linestyle=linetype)")
      diagrams.append("ax1.legend()")
      diagrams.append("ax2.plot(sim_res['ackF']/model_get('column.V'), sim_res['conductivity_detector.value'], color='m', linestyle=linetype)")      
      diagrams.append("ax3.step(sim_res['ackF']/model_get('column.V'), sim_res['tank_sample.Fsp'], color='g', linestyle=linetype)")     
      diagrams.append("ax4.plot(sim_res['ackF']/model_get('column.V'), sim_res['tank_buffer1.Fsp'], color='g', linestyle=linetype)")                
      diagrams.append("ax5.plot(sim_res['ackF']/model_get('column.V'), sim_res['tank_buffer2.Fsp'], color='g', linestyle=linetype)") 
      diagrams.append("ax6.plot(sim_res['ackF']/model_get('column.V'), sim_res['tank_harvest.V'], color='g', linestyle=linetype)") 


   elif plotType == 'Elution-conductivity-combined-all':
         
      # Part of plot made before simulation
      plt.figure()
      ax1 = plt.subplot(2,1,1)
      ax2 = plt.subplot(10,1,6)
      ax3 = plt.subplot(10,1,7)
      ax4 = plt.subplot(10,1,8)
      ax5 = plt.subplot(10,1,9)
      ax6 = plt.subplot(10,1,10)
 
      #ax1.set_title(title)
      ax1.grid()
      ax1.set_ylabel('c[P], c[A] [mg/mL]')

      ax2.grid()
      ax2.set_ylabel('c [mS/cm]')      

      ax3.grid()
      ax3.set_ylabel('F load [mL/min]')

      ax4.grid()
      ax4.set_ylabel('Fb1 [mL/min]')

      ax5.grid()
      ax5.set_ylabel('Fb2 [mL/min]')

      ax6.grid()
      ax6.set_ylabel('V [mL]')
      ax6.set_xlabel('Time [min] - relative start desorption')

      # Part of plot made after simulation
      diagrams.clear()    
      diagrams.append("ax1.plot(sim_res['time']-parDict['start_desorption']/model_get('control_buffer2.scaling'), \
                       sim_res['column.column_section[8].outlet.c[1]'], label='P', color='b', linestyle=linetype)")
      diagrams.append("ax1.plot(sim_res['time']-parDict['start_desorption']/model_get('control_buffer2.scaling'), \
                       sim_res['column.column_section[8].outlet.c[2]'], label='A', color='r', linestyle=linetype)")
      diagrams.append("ax1.legend()")
      
      diagrams.append("ax2.plot(sim_res['time']-parDict['start_desorption']/model_get('control_buffer2.scaling'), \
                       sim_res['conductivity_detector.value'], color='m', linestyle=linetype)")      
      diagrams.append("ax3.step(sim_res['time']-parDict['start_desorption']/model_get('control_buffer2.scaling'), \
                       sim_res['tank_sample.Fsp'], color='g', linestyle=linetype)")     
      diagrams.append("ax4.plot(sim_res['time']-parDict['start_desorption']/model_get('control_buffer2.scaling'), \
                       sim_res['tank_buffer1.Fsp'], color='g', linestyle=linetype)")                
      diagrams.append("ax5.plot(sim_res['time']-parDict['start_desorption']/model_get('control_buffer2.scaling'), \
                       sim_res['tank_buffer2.Fsp'], color='g', linestyle=linetype)") 
      diagrams.append("ax6.plot(sim_res['time']-parDict['start_desorption']/model_get('control_buffer2.scaling'), \
                       sim_res['tank_harvest.V'], color='g', linestyle=linetype)") 

   elif plotType == 'Elution-pooling':
         
      # Part of plot made before simulation
      plt.figure()
      ax1 = plt.subplot(3,1,1)
      ax2 = plt.subplot(3,1,2)
      ax3 = plt.subplot(6,1,5)
    
      ax1.set_title(title)
      ax1.grid()
      ax1.set_ylabel('c[P] and c[A]  [mg/mL]')
    
      ax2.grid()
      ax2.set_ylabel('c[P]+c[A] c[E]  [mg/mL]')
      
      ax3.grid()
      ax3.set_ylabel('Pooling [0/1]')
      ax3.set_xlabel('Time [min]')       

      # Part of plot made after simulation
      diagrams.clear()    
      diagrams.append("ax1.plot(sim_res['time'] - parDict['start_desorption']/model_get('control_buffer2.scaling'), \
                                sim_res['column.column_section[8].outlet.c[1]'], label='P', color='b', linestyle=linetype)")
      diagrams.append("ax1.plot(sim_res['time'] - parDict['start_desorption']/model_get('control_buffer2.scaling'), \
                                sim_res['column.column_section[8].outlet.c[2]'], label='A', color='r', linestyle=linetype)")
      diagrams.append("ax1.set_xlim(left=0)")
      diagrams.append("ax1.set_ylim([0,0.45])")
      diagrams.append("ax1.legend()")
 
      diagrams.append("ax2.plot(sim_res['time'] - parDict['start_desorption']/model_get('control_buffer2.scaling'), \
                                sim_res['uv_detector.value'], label='UV', color='k', linestyle=linetype)")
      diagrams.append("ax2.plot(sim_res['time'] - parDict['start_desorption']/model_get('control_buffer2.scaling'), \
                           0.05*sim_res['column.column_section[8].outlet.c[3]'], label='salt', color='m', linestyle=linetype)")
      diagrams.append("ax2.set_xlim(left=0)") 
      diagrams.append("ax2.set_ylim([0,0.45])")
      diagrams.append("ax2.legend()")
      
      diagrams.append("ax3.step(sim_res['time'] - parDict['start_desorption']/model_get('control_buffer2.scaling'), \
                                sim_res['control_pooling.out'], color='k', linestyle=linetype)")
      diagrams.append("ax3.set_xlim(left=0)")      

   elif plotType == 'Elution-vs-CV-pooling':
         
      # Part of plot made before simulation
      plt.figure()
      ax1 = plt.subplot(3,1,1)
      ax2 = plt.subplot(3,1,2)
      ax3 = plt.subplot(6,1,5)
    
      ax1.set_title(title)
      ax1.grid()
      ax1.set_ylabel('c[P] and c[A]  [mg/mL]')
    
      ax2.grid()
      ax2.set_ylabel('c[P]+c[A], c[E]  [mg/mL]')
      
      ax3.grid()
      ax3.set_ylabel('Pooling [0/1]')
      ax3.set_xlabel('Pumped liquid volume [CV]')

      # Part of plot made after simulation
      diagrams.clear()    
      diagrams.append("ax1.plot(sim_res['ackF']/model_get('column.V'), \
                                sim_res['column.column_section[8].outlet.c[1]'], label='P', color='b', linestyle=linetype)")
      diagrams.append("ax1.plot(sim_res['ackF']/model_get('column.V'), \
                                sim_res['column.column_section[8].outlet.c[2]'], label='A', color='r', linestyle=linetype)")
      diagrams.append("ax1.set_xlim(left=0)")
     # diagrams.append("ax1.set_ylim([0,0.45])")
      diagrams.append("ax1.legend()")
 
      diagrams.append("ax2.plot(sim_res['ackF']/model_get('column.V'), \
                                sim_res['uv_detector.value'], label='UV', color='k', linestyle=linetype)")
      diagrams.append("ax2.plot(sim_res['ackF']/model_get('column.V'), \
                           0.05*sim_res['column.column_section[8].outlet.c[3]'], label='salt', color='m', linestyle=linetype)")
      diagrams.append("ax2.set_xlim(left=0)") 
    # diagrams.append("ax2.set_ylim([0,0.45])")
      diagrams.append("ax2.legend()")
      
      diagrams.append("ax3.step(sim_res['ackF']/model_get('column.V'), \
                                sim_res['control_pooling.out'], color='k', linestyle=linetype)")
      diagrams.append("ax3.set_xlim(left=0)")      


   elif plotType == 'Pooling':
      
      # Part of plot made before simulation
      plt.figure()
      ax1 = plt.subplot(3,1,1)
      ax2 = plt.subplot(3,1,2)
      ax3 = plt.subplot(3,1,3)
    
      ax1.set_title(title)
      ax1.grid()
      ax1.set_ylabel('m[P], m[A] - harvest  [mg]')
          
      ax2.grid()
      ax2.set_ylabel('m[P], m[A] - waste  [mg]')
    
      ax3.grid()
      ax3.set_ylabel('Pooling [0/1]')
      ax3.set_xlabel('Time [min]')       

      # Part of plot made after simulation
      diagrams.clear()    
      diagrams.append("ax1.plot(sim_res['time'], sim_res['tank_harvest.m[1]'], label='P', color='b', linestyle=linetype)")
      diagrams.append("ax1.plot(sim_res['time'], sim_res['tank_harvest.m[2]'], label='A', color='r', linestyle=linetype)")
      diagrams.append("ax1.legend()")

      diagrams.append("ax2.plot(sim_res['time'], sim_res['tank_waste.m[1]'], label='P', color='b', linestyle=linetype)")
      diagrams.append("ax2.plot(sim_res['time'], sim_res['tank_waste.m[2]'], label='A', color='r', linestyle=linetype)")
      diagrams.append("ax2.legend()")
       
      diagrams.append("ax3.step(sim_res['time'], sim_res['control_pooling.out'], color='k', linestyle=linetype)")

   elif plotType == 'Column-outlet':
         
      # Part of plot made before simulation
      plt.figure()
      ax1 = plt.subplot(3,1,1)
      ax2 = plt.subplot(3,1,2)
      ax3 = plt.subplot(3,1,3)
    
      ax1.set_title(title)
      ax1.grid()
      ax1.set_ylabel('c[P]')
          
      ax2.grid()
      ax2.set_ylabel('c[A]')
    
      ax3.grid()
      ax3.set_ylabel('c[E]')
      ax3.set_xlabel('Time [min]')       

      # Part of plot made after simulation
      diagrams.clear()    
      diagrams.append("ax1.plot(sim_res['time'], sim_res['column.outlet.c[1]'], label='P', color='b', linestyle=linetype)")
      diagrams.append("ax2.plot(sim_res['time'], sim_res['column.outlet.c[2]'], label='P', color='b', linestyle=linetype)")
      diagrams.append("ax3.plot(sim_res['time'], sim_res['column.outlet.c[3]'], label='A', color='r', linestyle=linetype)")

   else:
      print("Plot window type not correct") 

# Define and extend describe for the current application
def describe(name, decimals=3):
   """Look up description of culture, media, as well as parameters and variables in the model code"""

   if name == 'chromatography':
      print('Ion exchange chromatorgraphy controlled with varying salt-concentration. The pH is kept constant.')        

   elif name in ['liquidphase', 'media']:
      P = model_get('liquidphase.P'); P_description = model_get_variable_description('liquidphase.P'); 
      P_mw = model_get('liquidphase.mw[1]')
      A = model_get('liquidphase.A'); A_description = model_get_variable_description('liquidphase.A'); 
      A_mw = model_get('liquidphase.mw[2]')
      E = model_get('liquidphase.E'); E_description = model_get_variable_description('liquidphase.E'); 
      E_mw = model_get('liquidphase.mw[3]')
      PS = model_get('liquidphase.PS'); PS_description = model_get_variable_description('liquidphase.PS'); 
      PS_mw = model_get('liquidphase.mw[4]')
      AS = model_get('liquidphase.AS'); AS_description = model_get_variable_description('liquidphase.AS'); 
      AS_mw = model_get('liquidphase.mw[5]')

      print('Chromatography liquidphase (or mobilephase) substances included in the model')
      print()
      print(P_description, '                 - index = ', P, '- molecular weight = ', P_mw, 'Da')
      print(A_description, '      - index = ', A, '- molecular weight = ', A_mw, 'Da')
      print(E_description, '                     - index = ', E, '- molecular weight = ', E_mw, 'Da')
      print(PS_description, '           - index = ', PS, '- molecular weight = ', PS_mw, 'Da')
      print(AS_description, '- index = ', AS, '- molecular weight = ', AS_mw, 'Da')
      print()
      print('Note that both proteins P and A as well as the salt-ion E is modelled to the same mobile phase volume.')

   elif name in ['parts']:
      describe_parts(component_list_minimum)
      
   elif name in ['MSL']:
      describe_MSL()

   else:
      describe_general(name, decimals)
         
#------------------------------------------------------------------------------------------------------------------
#  General code 
FMU_explore = 'FMU-explore for FMPy version 0.9.8'
#------------------------------------------------------------------------------------------------------------------

# Define function par() for parameter update
def par(parDict=parDict, parCheck=parCheck, parLocation=parLocation, *x, **x_kwarg):
   """ Set parameter values if available in the predefined dictionaryt parDict. """
   x_kwarg.update(*x)
   x_temp = {}
   for key in x_kwarg.keys():
      if key in parDict.keys():
         x_temp.update({key: x_kwarg[key]})
      else:
         print('Error:', key, '- seems not an accessible parameter - check the spelling')
   parDict.update(x_temp)
   
   parErrors = [requirement for requirement in parCheck if not(eval(requirement))]
   if not parErrors == []:
      print('Error - the following requirements do not hold:')
      for index, item in enumerate(parErrors): print(item)

# Define function init() for initial values update
def init(parDict=parDict, *x, **x_kwarg):
   """ Set initial values and the name should contain string '_0' to be accepted.
       The function can handle general parameter string location names if entered as a dictionary. """
   x_kwarg.update(*x)
   x_init={}
   for key in x_kwarg.keys():
      if '_0' in key: 
         x_init.update({key: x_kwarg[key]})
      else:
         print('Error:', key, '- seems not an initial value, use par() instead - check the spelling')
   parDict.update(x_init)

# Define fuctions similar to pyfmi model.get(), model.get_variable_descirption(), model.get_variable_unit()
def model_get(parLoc, model_description=model_description):
   """ Function corresponds to pyfmi model.get() but returns just a value and not a list"""
   par_var = model_description.modelVariables
   for k in range(len(par_var)):
      if par_var[k].name == parLoc:
         try:
            if par_var[k].name in start_values.keys():
                  value = start_values[par_var[k].name]
            elif par_var[k].variability in ['constant']:        
                  value = float(par_var[k].start)                          
            elif par_var[k].variability in ['fixed', 'continuous']:
               try:
                  value = sim_res[par_var[k].name][-1]
               except (AttributeError, ValueError):
                  value = None
                  print('Variable not logged')
            else:
               value = None
         except NameError:
            print('Error: Information available after first simulation')
            value = None
   return value

def model_get_variable_description(parLoc, model_description=model_description):
   """ Function corresponds to pyfmi model.get_variable_description() but returns just a value and not a list"""
   par_var = model_description.modelVariables
#   value = [x[1] for x in [(par_var[k].name, par_var[k].description) for k in range(len(par_var))] if parLoc in x[0]]
   value = [x.description for x in par_var if parLoc in x.name]   
   return value[0]
   
def model_get_variable_unit(parLoc, model_description=model_description):
   """ Function corresponds to pyfmi model.get_variable_unit() but returns just a value and not a list"""
   par_var = model_description.modelVariables
#   value = [x[1] for x in [(par_var[k].name, par_var[k].unit) for k in range(len(par_var))] if parLoc in x[0]]
   value = [x.unit for x in par_var if parLoc in x.name]
   return value[0]
      
# Define function disp() for display of initial values and parameters
def disp(name='', decimals=3, mode='short'):
   """ Display intial values and parameters in the model that include "name" and is in parLocation list.
       Note, it does not take the value from the dictionary par but from the model. """
   
   def dict_reverser(d):
      seen = set()
      return {v: k for k, v in d.items() if v not in seen or seen.add(v)}
   
   if mode in ['short']:
      k = 0
      for Location in [parLocation[k] for k in parDict.keys()]:
         if name in Location:
            if type(model_get(Location)) != np.bool_:
               print(dict_reverser(parLocation)[Location] , ':', np.round(model_get(Location),decimals))
            else:
               print(dict_reverser(parLocation)[Location] , ':', model_get(Location))               
         else:
            k = k+1
      if k == len(parLocation):
         for parName in parDict.keys():
            if name in parName:
               if type(model_get(Location)) != np.bool_:
                  print(parName,':', np.round(model_get(parLocation[parName]),decimals))
               else: 
                  print(parName,':', model_get(parLocation[parName])[0])

   if mode in ['long','location']:
      k = 0
      for Location in [parLocation[k] for k in parDict.keys()]:
         if name in Location:
            if type(model_get(Location)) != np.bool_:       
               print(Location,':', dict_reverser(parLocation)[Location] , ':', np.round(model_get(Location),decimals))
         else:
            k = k+1
      if k == len(parLocation):
         for parName in parDict.keys():
            if name in parName:
               if type(model_get(Location)) != np.bool_:
                  print(parLocation[parName], ':', dict_reverser(parLocation)[Location], ':', parName,':', 
                     np.round(model_get(parLocation[parName]),decimals))

# Line types
def setLines(lines=['-','--',':','-.']):
   """Set list of linetypes used in plots"""
   global linecycler
   linecycler = cycle(lines)

# Show plots from sim_res, just that
def show(diagrams=diagrams):
   """Show diagrams chosen by newplot()"""
   # Plot pen
   linetype = next(linecycler)    
   # Plot diagrams 
   for command in diagrams: eval(command)

# Define simulation
def simu(simulationTime=simulationTime, mode='Initial', options=opts_std, diagrams=diagrams):
   """Model loaded and given intial values and parameter before, and plot window also setup before."""   
   
   # Global variables
   global sim_res, prevFinalTime, stateDict, stateDictInitial, stateDictInitialLoc, start_values
   
   # Simulation flag
   simulationDone = False
   
   # Internal help function to extract variables to be stored
   def extract_variables(diagrams):
       output = []
       variables = [v for v in model_description.modelVariables if v.causality == 'local']
       for j in range(len(diagrams)):
           for k in range(len(variables)):
               if variables[k].name in diagrams[j]:
                   output.append(variables[k].name)
       return output

   # Run simulation
   if mode in ['Initial', 'initial', 'init']: 
      
      start_values = {parLocation[k]:parDict[k] for k in parDict.keys()}
      
      # Simulate
      sim_res = simulate_fmu(
         filename = fmu_model,
         validate = False,
         start_time = 0,
         stop_time = simulationTime,
         output_interval = simulationTime/options['NCP'],
         record_events = True,
         start_values = start_values,
         fmi_call_logger = None,
         output = list(set(extract_variables(diagrams) + list(stateDict.keys()) + key_variables))
      )
      
      simulationDone = True
      
   elif mode in ['Continued', 'continued', 'cont']:
      
      if prevFinalTime == 0: 
         print("Error: Simulation is first done with default mode = init'")
         
      else:         
         # Update parDictMod and create parLocationMod
         parDictRed = parDict.copy()
         parLocationRed = parLocation.copy()
         for key in parDict.keys():
            if parLocation[key] in stateDictInitial.values(): 
               del parDictRed[key]  
               del parLocationRed[key]
         parLocationMod = dict(list(parLocationRed.items()) + list(stateDictInitialLoc.items()))
   
         # Create parDictMod and parLocationMod
         parDictMod = dict(list(parDictRed.items()) + 
            [(stateDictInitial[key], stateDict[key]) for key in stateDict.keys()])      

         start_values = {parLocationMod[k]:parDictMod[k] for k in parDictMod.keys()}
  
         # Simulate
         sim_res = simulate_fmu(
            filename = fmu_model,
            validate = False,
            start_time = prevFinalTime,
            stop_time = prevFinalTime + simulationTime,
            output_interval = simulationTime/options['NCP'],
            record_events = True,
            start_values = start_values,
            fmi_call_logger = None,
            output = list(set(extract_variables(diagrams) + list(stateDict.keys()) + key_variables))
         )
      
         simulationDone = True
   else:
      
      print("Error: Simulation mode not correct")

   if simulationDone:
      
      # Plot diagrams from simulation
      linetype = next(linecycler)    
      for command in diagrams: eval(command)
   
      # Store final state values in stateDict:        
      for key in stateDict.keys(): stateDict[key] = model_get(key)  
         
      # Store time from where simulation will start next time
      prevFinalTime = sim_res['time'][-1]
      
   else:
      print('Error: No simulation done')
            
# Describe model parts of the combined system
def describe_parts(component_list=[]):
   """List all parts of the model""" 
       
   def model_component(variable_name):
      i = 0
      name = ''
      finished = False
      if not variable_name[0] == '_':
         while not finished:
            name = name + variable_name[i]
            if i == len(variable_name)-1:
                finished = True 
            elif variable_name[i+1] in ['.', '(']: 
                finished = True
            else: 
                i=i+1
      if name in ['der', 'temp_1', 'temp_2', 'temp_3', 'temp_4', 'temp_5', 'temp_6', 'temp_7']: name = ''
      return name
    
#   variables = list(model.get_model_variables().keys())
   variables = [v.name for v in model_description.modelVariables]
        
   for i in range(len(variables)):
      component = model_component(variables[i])
      if (component not in component_list) \
      & (component not in ['','BPL', 'Customer', 'today[1]', 'today[2]', 'today[3]', 'temp_2', 'temp_3']):
         component_list.append(component)
      
   print(sorted(component_list, key=str.casefold))

# Describe MSL   
def describe_MSL(flag_vendor=flag_vendor):
   """List MSL version and components used"""
   print('MSL:', MSL_usage)
 
# Describe parameters and variables in the Modelica code
def describe_general(name, decimals):
  
   if name == 'time':
      description = 'Time'
      unit = 'h'
      print(description,'[',unit,']')
      
   elif name in parLocation.keys():
      description = model_get_variable_description(parLocation[name])
      value = model_get(parLocation[name])
      try:
         unit = model_get_variable_unit(parLocation[name])
      except FMUException:
         unit =''
      if unit =='':
         if type(value) != np.bool_:
            print(description, ':', np.round(value, decimals))
         else:
            print(description, ':', value)            
      else:
        print(description, ':', np.round(value, decimals), '[',unit,']')
                  
   else:
      description = model_get_variable_description(name)
      value = model_get(name)
      try:
         unit = model_get_variable_unit(name)
      except FMUException:
         unit =''
      if unit =='':
         if type(value) != np.bool_:
            print(description, ':', np.round(value, decimals))
         else:
            print(description, ':', value)     
      else:
         print(description, ':', np.round(value, decimals), '[',unit,']')
         
# Describe framework
def BPL_info():
   print()
   print('Model for bioreactor has been setup. Key commands:')
   print(' - par()       - change of parameters and initial values')
   print(' - init()      - change initial values only')
   print(' - simu()      - simulate and plot')
   print(' - newplot()   - make a new plot')
   print(' - show()      - show plot from previous simulation')
   print(' - disp()      - display parameters and initial values from the last simulation')
   print(' - describe()  - describe culture, broth, parameters, variables with values/units')
   print()
   print('Note that both disp() and describe() takes values from the last simulation')
   print()
   print('Brief information about a command by help(), eg help(simu)') 
   print('Key system information is listed with the command system_info()')

def system_info():
   """Print system information"""
#   FMU_type = model.__class__.__name__
   constants = [v for v in model_description.modelVariables if v.causality == 'local']
   
   print()
   print('System information')
   print(' -OS:', platform.system())
   print(' -Python:', platform.python_version())
   try:
       scipy_ver = scipy.__version__
       print(' -Scipy:',scipy_ver)
   except NameError:
       print(' -Scipy: not installed in the notebook')
   print(' -FMPy:', version('fmpy'))
   print(' -FMU by:', read_model_description(fmu_model).generationTool)
   print(' -FMI:', read_model_description(fmu_model).fmiVersion)
   if model_description.modelExchange is None:
      print(' -Type: CS')
   else:
      print(' -Type: ME')
   print(' -Name:', read_model_description(fmu_model).modelName)
   print(' -Generated:', read_model_description(fmu_model).generationDateAndTime)
   print(' -MSL:', MSL_version)    
   print(' -Description:', BPL_version)   
   print(' -Interaction:', FMU_explore)
   
#------------------------------------------------------------------------------------------------------------------
#  Startup
#------------------------------------------------------------------------------------------------------------------

BPL_info()