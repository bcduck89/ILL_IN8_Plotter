import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import os
from datetime import datetime, timedelta
import re

from scipy import sparse
from scipy.sparse.linalg import spsolve
import scipy.interpolate as interpolate
from scipy.signal import find_peaks

from pybaselines import Baseline, utils

from lmfit import Parameters, Minimizer
from lmfit.models import GaussianModel, ConstantModel, PolynomialModel, ExponentialModel, LinearModel

# ILL law data 로딩
def data_loader(file_path):
    with open(file_path, 'r') as f:
        data_law = f.readlines()
        
        # 데이터 부분만 추출
    for idx, line in enumerate(data_law):
        striped_line = line.strip()
        
        # 커맨드라인에서 데이터 포인트가 몇개인지 확인하기
        if 'COMND' in striped_line:
            np_idx = striped_line.split(' ').index('np')
            np = int(striped_line.split(' ')[np_idx + 1])
        
        # 데이터 시작 부터 데이터 포인터 개수 + 2만큼(컬럼 포함) 가져오기 
        if 'DATA' in striped_line:
            data_tmp1 = data_law[idx + 1 : idx + np + 2]
            data_header = data_law[: idx]
              
            # 데이터 부분 데이터 프레임으로 가공
            data_only = []
            for line in data_tmp1:
                striped_line = line.strip()
                refined_line = [element for element in striped_line.split(' ') if element != '']
                data_only.append(refined_line)
            
            data_only = pd.DataFrame(data_only[1:], columns=data_only[0]).astype(float)    
        
    return data_law, data_header, data_only

def data_main_info_extract(data_header):

    # 데이터 확인하기 - header
    # data_header
    
    # 데이터 확인하기 - data_only 5줄만
    # data_only.head()
    
    # 파일별로 중요한 데이터 정보 확인하기
    # ['FILE_', 'DATE_','COMND']
    file = [line.strip().split(': ')[-1] for line in data_header if 'FILE_' in line]
    date = [line.strip().split(': ')[-1] for line in data_header if 'DATE_' in line]
    comnd = [line.strip().split(': ')[-1] for line in data_header if 'COMND' in line]
    data_info_per_file = {'file_name': file,
                          'measure_date': date,
                          'command': comnd}
    
    # 데이터 확인하기 - 데이터 중요 정보
    # data_info_per_file
    return data_info_per_file

def make_data_info_table(file_name_list, data_dir, save = False):

    # # 데이터 확인하기
    # file_name_list
    
    # 데이터 저장할 디렉토리
    root_dir = os.getcwd()
    save_dir = root_dir
    save_file_name = data_dir.split('/')[-1]
    save_path = f'{save_dir}/{save_file_name}.csv'
    
    # ========================================================================
    # 데이터 담을 그릇 준비
    data_info_df = pd.DataFrame(columns = ['file_name', 'num_data_measured', 'measure_date', 'measure_time(min)', 'command'])
    
    # 데이터 디렉토리에 있는 모든 파일 불러오기
    for file_name in file_name_list:
        # 데이터 1개 경로
        file_path = f'{data_dir}/{file_name}'
    
        # 데이터 1개 불러오기
        _, data_header, data_only = data_loader(file_path)
        
        # 데이터 핵심 정보 추출하기
        data_info_per_file = data_main_info_extract(data_header)

        # 실제 데이터 포인트 측정 수 추출하기
        num_data_measured = data_only.shape[0]
        
        # 데이터 측정 시간 추출하기
        measure_time = np.round(data_only['TIME'].sum()/60 , 1) # 단위 : 분
        data_info_per_file['measure_time(min)'] = np.round(measure_time, 1)
        data_info_per_file['num_data_measured'] = num_data_measured
        
        # 데이터 담기
        data_info_df = pd.concat([data_info_df, pd.DataFrame(data_info_per_file, index=[0])], ignore_index=True)

    # ========================================================================
    # 파일 1개당 측정하는데 걸리는 총 시간 계산
    measure_delta_time = (pd.Series(pd.to_datetime(data_info_df['measure_date'].to_list()[1:]) -
     pd.to_datetime(data_info_df['measure_date'].to_list()[:-1])).dt.total_seconds()/60).round(1).to_list() + [None]
    
    measure_delta_time_filtered = []
    # 첫번째 실험과 두번째 실험 사이에 1달정도의 시간 차이가 있음. 이 부분을 필터링 하기 위한 코드
    for idx, delta_time in enumerate(measure_delta_time):
        if delta_time == None:
            measure_delta_time_filtered.append(0)
        else:
            if delta_time >= timedelta(minutes = 1440).total_seconds()/60: # 1일보다 길어지면
                measure_delta_time_filtered.append(0)
            else:
                measure_delta_time_filtered.append(delta_time)
                
    data_info_df.insert(loc = 3, value = measure_delta_time_filtered, column = 'measure_delta_time(min)')

    # ========================================================================
    # 한 Q포인트당 장비가 셋팅 되는데 걸리는 시간 계산
    instrument_setting_time = np.round(data_info_df['measure_delta_time(min)'] - data_info_df['measure_time(min)'], 1)
    for inst_set_time in instrument_setting_time:
        if inst_set_time < 0:
            negative_value_idx = instrument_setting_time[(instrument_setting_time == inst_set_time)].index
            instrument_setting_time.iloc[negative_value_idx] = 0
    
    data_info_df.insert(loc = 4, value = instrument_setting_time, column = 'instrument_setting_time(min)')

    # ========================================================================
    # 정규식 패턴 (숫자와 문자+숫자)
    pattern = re.compile(r'[a-zA-Z][a-zA-Z]|[a-zA-Z]\d')
    
    # 측정 데이터는 정규식 패턴 속에서 bs와 qh가 command에 포함되어있음
    sort_list = []
    for command_line in data_info_df['command']:
        re_result = pattern.findall(command_line)
        
        # 측정데이터 조건A
        if re_result[0] == 'bs' and re_result[1] == 'qh':
            sort_list.append('measure')
        else:
            sort_list.append('align')

    data_info_df.insert(loc = 1, value = sort_list, column = 'align/measure')
    # ========================================================================
    # 데이터 추가 정보 저장하기
    hkl_list = []
    energy_start_list = []
    energy_step_list = []
    energy_end_list = []
    num_data_command_list = []
    mn_list = []
    for a_or_m, command_line in zip(data_info_df['align/measure'], data_info_df['command']):
        if a_or_m == 'align':
            hkl_list.append('')
            energy_start_list.append('')
            energy_step_list.append('')
            energy_end_list.append('')
            num_data_command_list.append('')
            mn_list.append('')
        
        elif a_or_m == 'measure':
            h = np.round(float(command_line.split(' ')[2:5][0]),1)
            k = np.round(float(command_line.split(' ')[2:5][1]),1)
            l = np.round(float(command_line.split(' ')[2:5][2]),1)            
            energy_start = float(command_line.split(' ')[5])
            energy_step = float(command_line.split(' ')[10])
            num_data_command = int(command_line.split(' ')[12])
            energy_end = energy_start + energy_step*(num_data_command-1)
            mn = command_line.split(' ')[14]

            hkl_list.append(f'{h} {k} {l}')
            energy_start_list.append(energy_start)
            energy_step_list.append(energy_step)
            energy_end_list.append(energy_end)
            num_data_command_list.append(num_data_command)
            mn_list.append(mn)
    
    data_info_df.insert(loc = 2, value = hkl_list, column = 'H K L')
    data_info_df.insert(loc = 3, value = energy_start_list, column = 'energy_start')
    data_info_df.insert(loc = 4, value = energy_end_list, column = 'energy_end')    
    data_info_df.insert(loc = 5, value = energy_step_list, column = 'energy_step')
    data_info_df.insert(loc = 6, value = num_data_command_list, column = 'num_data_command')
    data_info_df.insert(loc = 8, value = mn_list, column = 'mon')      


    # 데이터 수정 코드1: 측정 데이터 개수 -> 얼라인 파일 경우, 생략
    align_files_idx = data_info_df[(data_info_df['align/measure'] == 'align')].index
    data_info_df['num_data_measured'].iloc[align_files_idx] = ''
    
    
    # ========================================================================
    # 데이터 저장 코드
    if save == True:
        # 저장할 파일 경로 설정
        counter = 1
        while True:
            if counter > 1:
                # 파일 이름에 숫자를 붙여줍니다 (예: 파일명 (1).csv, 파일명 (2).csv, ...)
                save_path = f'{save_dir}/{save_file_name} ({counter}).csv'
            if not os.path.exists(save_path):
                break
            counter += 1
            
        data_info_df.to_csv(save_path)
    elif save == False:
        pass

    return data_info_df


def seperate_align_measure_data(data_info_df):
    # 정규식 패턴 (숫자와 문자+숫자)
    pattern = re.compile(r'[a-zA-Z][a-zA-Z]|[a-zA-Z]\d')
    
    # 측정 데이터는 정규식 패턴 속에서 bs와 qh가 command에 포함되어있음
    condition_bs = data_info_df['command'].apply(lambda x : pattern.findall(x)).str.get(0) == 'bs'
    condition_qh = data_info_df['command'].apply(lambda x : pattern.findall(x)).str.get(1) == 'qh'
    condition_measure_data = condition_bs * condition_qh
    condition_align_data = ~condition_measure_data
    
    data_info_measure_df = data_info_df[condition_measure_data]
    data_info_align_df = data_info_df[condition_align_data]
    
    return data_info_measure_df, data_info_align_df

def interpolate_energy_cnts(energy, cnts, inc_ratio = 5, energy_step = 0.5):
    # interpolating
    # 데이터 증가 비율
    
    data_len_new = (len(energy) - 1) * inc_ratio + 1
    energy_new = np.round(np.arange(data_len_new) * energy_step/inc_ratio + energy.min(), 1)
    cnts_func = interpolate.interp1d(energy, cnts)
    cnts_new = cnts_func(energy_new)
    
    # interpolation 후, 데이터 수 확인
    print(f'energy_new: {len(energy_new)}, cnts_new: {len(cnts_new)}')

    return energy_new, cnts_new

def remove_baseline(energy_new, cnts_new):
    # 베이스 라인 제거하기
    baseline_fitter = Baseline(x_data=energy_new)
    regular_asls = baseline_fitter.modpoly(cnts_new, poly_order=1)[0]
    cnts_baselined = cnts_new - regular_asls
    
    regular_asls -= np.abs(cnts_baselined.min())
    cnts_baselined += np.abs(cnts_baselined.min())
    
    return regular_asls, cnts_baselined

def fitting_single_data(energy, cnts, energy_new, cnts_baselined, peak_energy_list = []):
    
    if peak_energy_list == []:
        peak_energy_list = np.random.choice(energy, size = 5)
    else:
        pass
        
    peak_idx_list = []
    for peak_energy in peak_energy_list:
        peak_idx = np.where(energy == peak_energy)[0][0]
        peak_idx_list.append(peak_idx)
    
    # 모델 생성
    model = ConstantModel()
    
    # 파라미터 생성
    params = model.make_params()
    
    # 컨스턴트 파라미터 초기값 설정
    params['c'].set(min = 0, vary = False )
    
    # peak 모델 생성(Gaussian)
    model_list = []
    param_list = []
    for k, peak_position in enumerate(peak_idx_list):
    
        g = GaussianModel(prefix=f'g{k+1}_')
        p = g.make_params()
        p[f'g{k+1}_center'].set(energy[peak_position], min = energy[peak_position] - 3,  max = energy[peak_position] + 3) 
        p[f'g{k+1}_height'].set(0, min = 0)
        p[f'g{k+1}_amplitude'].set(cnts[peak_position], min = 0) 
        p[f'g{k+1}_sigma'].set(0.1, vary = True)
        
        model_list.append(g)
        param_list.append(p)
    
    for model_ in model_list:
        model += model_
    
    for param in param_list:
        params.update(param)
    
    result = model.fit(data = cnts_baselined, params=params, x = energy_new )
    components = result.eval_components()

    return result, components

# ======================================================================
def plot_single_plot(file_name, data_only, data_info_per_file, peak_energy_list):

    # 데이터 가공하기 및 계산하기
    
    energy = data_only['EN'].round(1)
    mn = data_only['M1']
    cnts = data_only['CNTS']
    
    # 두개의 푸아송 분포 표준편차의 합의 표준편차
    poisson_std = np.sqrt(cnts)
    
    # errorbar
    cnts_error= [poisson_std, poisson_std]
    # ======================================================================
    # interpolation
    energy_new, cnts_new = interpolate_energy_cnts(energy, cnts, inc_ratio = 5, energy_step = 0.5)
    
    # ======================================================================
    # 베이스 라인 제거하기
    regular_asls, cnts_baselined = remove_baseline(energy_new, cnts_new)
    index_cnts = np.unique(np.arange(len(energy_new))//5) * 5
    
    # ======================================================================
    # fitting
    result, components = fitting_single_data(energy, cnts, energy_new, cnts_baselined, peak_energy_list = peak_energy_list)
    cnts_fitted = result.best_fit
    
    # =====================================================================
    # 데이터 그리기
    fig, axs = plt.subplots(nrows = 1, ncols = 2, figsize = (16, 6))
    bbox_to_anchor = (0.5, -0.3)
    
    axs[0].errorbar(energy, cnts, yerr =  cnts_error, capsize=3, color = 'k', ecolor = 'k', linestyle = '--', label = data_info_per_file['file_name'][0])
    axs[0].scatter(energy, cnts, s = 50, color = 'k')
    axs[0].plot(energy_new, regular_asls, color = 'k', linestyle = '--')
    
    axs[0].errorbar(energy, cnts_baselined[index_cnts], yerr =  cnts_error, capsize=3, color = 'b', ecolor = 'k', linestyle = '--', label = f"{data_info_per_file['file_name'][0]}_baselined")
    axs[0].scatter(energy, cnts_baselined[index_cnts] , s = 50, color = 'b')
    
    xticks = np.arange(energy.min(), energy.max()+1, 1)
    axs[0].set_xticks(xticks)
    axs[0].legend(loc = 'lower center', bbox_to_anchor=(0.5, -0.3), ncols = 5,  fontsize = 10)
    axs[0].grid(True)
    
    axs[1].errorbar(energy, cnts_baselined[index_cnts], yerr =  cnts_error, capsize=3, color = 'k', ecolor = 'k', linestyle = '--', label = f"{data_info_per_file['file_name'][0]}_baselined")
    axs[1].scatter(energy, cnts_baselined[index_cnts] , s = 50, color = 'k')
    axs[1].plot(energy_new, cnts_fitted, color = 'r')
    
    peak_energy_fit_dict = {}
    for n, (name, component) in enumerate(components.items()):
        if name == 'constant':
            axs[1].plot(energy_new, component, '--', label = name, color = 'k')
        else:
            center = result.params[f'{name}center'].value
            fwhm = result.params[f'{name}fwhm'].value
            axs[1].plot(energy_new, component, '--', label = name,)
            
            peak_energy_fit_dict[name] = [center, fwhm]
            
            if center -fwhm/2 <= energy.min() or center +fwhm/2 >= energy.max():
                pass
            elif center >= energy.min() or center <= energy.max():
                axs[1].hlines([component.max()/2], xmin = center - fwhm/2, xmax = center + fwhm/2, colors = 'k')
                axs[1].vlines([center], ymin = -cnts_baselined[index_cnts,].max() * 0.2, ymax = 0, colors = 'g', linewidth = 3)
                # axs[1].annotate(f'{name}\n{round(center,2)}', xy = [center - 0.5*0.2, -cnts_fitted.max() * 0.2 * 1.2])
            else:
                pass
    
    axs[1].set_xticks(xticks)
    axs[1].legend(loc = 'lower center', bbox_to_anchor=(0.5, -0.3), ncols = 5,  fontsize = 10)
    axs[1].grid(True)
    
    
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.2, hspace=0.4)

    return energy, cnts, energy_new, cnts_new, regular_asls, cnts_baselined, peak_energy_fit_dict, indexs_cnts, result, components

def plot_substract_noise_data(energy, cnt_fitted, peak_all_list, peak_real_name_list, peak_noise_name_list, index_cnts):
    # ==============================================================================================
    # noise peak을 fit curve에서 빼기
    substrated_result = cnts_fitted.copy()
    for peak_noise_name in peak_noise_name_list:
        peak_noise_component = components[peak_noise_name]
        substrated_result -= peak_noise_component
    
    fig, axs = plt.subplots(nrows = 1, ncols = 1, figsize = (8, 6))
    bbox_to_anchor = (0.5, -0.3)
    
    # real peak과 noise peak이 빠진 커브를 새로 그리기
    for peak_real_name in peak_real_name_list:
        peak_real_component = components[peak_real_name]
    
        center = peak_energy_fit_dict[peak_real_name][0]
        fwhm = peak_energy_fit_dict[peak_real_name][1]
        axs.plot(energy_new, peak_real_component, '--', label = peak_real_name)
    
        if center - fwhm/2 <= energy.min() or center + fwhm/2 >= energy.max():
            pass
        elif center >= energy.min() or center <= energy.max():
            axs.vlines([center], ymin = -substrated_result.max() * 0.2, ymax = 0, colors = 'g', linewidth = 3)
            axs.hlines([peak_real_component.max()/2], xmin = center - fwhm/2, xmax = center + fwhm/2, colors = 'k')
            axs.annotate(f'{peak_real_name}\n{round(center,2)}', xy = [center - 0.5*0.2, -substrated_result.max() * 0.2 * 1.2])
        else:
            pass
    
    axs.scatter(energy, substrated_result[index_cnts], color = 'k', linestyle = '--', label = f"{data_info_per_file['file_name'][0]}_baselined_sub_noise")
    axs.plot(energy_new, substrated_result, color = 'r')
    
    xticks = np.arange(energy.min(), energy.max()+1, 1)
    axs.set_xticks(xticks)
    axs.legend(loc = 'lower center', bbox_to_anchor=(0.5, -0.3), ncols = 5,  fontsize = 10)
    axs.grid(True)

    return substrated_result
