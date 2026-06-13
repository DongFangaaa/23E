#include "stm32f10x.h"                  // Device header
#include "delay.h"

void HCSR04_Init()
{
	RCC_APB2PeriphClockCmd(RCC_APB2Periph_GPIOB,ENABLE);
	RCC_APB1PeriphClockCmd(RCC_APB1Periph_TIM3,ENABLE);
	
	GPIO_InitTypeDef GPIO_Initstructure;
	GPIO_Initstructure.GPIO_Mode = GPIO_Mode_Out_PP;
	GPIO_Initstructure.GPIO_Pin = GPIO_Pin_10;
	GPIO_Initstructure.GPIO_Speed = GPIO_Speed_50MHz;
	GPIO_Init(GPIOB,&GPIO_Initstructure);
	
	GPIO_Initstructure.GPIO_Mode = GPIO_Mode_IPD;
	GPIO_Initstructure.GPIO_Pin = GPIO_Pin_11;
	GPIO_Init(GPIOB,&GPIO_Initstructure);
	
	TIM_InternalClockConfig(TIM3);
	
	TIM_TimeBaseInitTypeDef TIM_TimeBaseStructure;
  TIM_TimeBaseStructure.TIM_Period = 60000 - 1; 
  TIM_TimeBaseStructure.TIM_Prescaler = 72 - 1;//一次1us
  TIM_TimeBaseStructure.TIM_ClockDivision = TIM_CKD_DIV1;
  TIM_TimeBaseStructure.TIM_CounterMode = TIM_CounterMode_Up;
  TIM_TimeBaseInit(TIM3, &TIM_TimeBaseStructure);
	
	//TIM_Cmd(TIM3,ENABLE);
}

float HCSR04_GetValue()
{
	uint16_t time = 0;
	uint32_t timeout = 1000000;
    
  GPIO_SetBits(GPIOB, GPIO_Pin_10); // Trig 发送高电平
  Delay_us(20);
  GPIO_ResetBits(GPIOB, GPIO_Pin_10);

  while (GPIO_ReadInputDataBit( GPIOB, GPIO_Pin_11	) == 0)
		{
        timeout--;
        if (timeout == 0) return -1.0f; // 传感器无响应，直接返回错误码
    }	// 等待变高
		
  TIM_SetCounter(TIM3, 0); // 清零
  TIM_Cmd(TIM3, ENABLE);   // 开始计时

	timeout = 1000000;
  while (GPIO_ReadInputDataBit(GPIOB, GPIO_Pin_11) == 1)
		{
        timeout--;
        if (timeout == 0) 
        {
            TIM_Cmd(TIM3, DISABLE);
            return -2.0f; // 测距超时
        }
    }		// 等待变低
  TIM_Cmd(TIM3, DISABLE);  // 停止计时
    
  time = TIM_GetCounter(TIM3); // us计数值
    
  return (float)time * 0.017f; // 距离
}
