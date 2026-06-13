#include "stm32f10x.h"                  // Device header

volatile uint32_t Timer_Tick = 0;

void Timer_Init()
{
	RCC_APB2PeriphClockCmd(RCC_APB2Periph_TIM1,ENABLE);
	
	TIM_InternalClockConfig(TIM1);
	
	TIM_TimeBaseInitTypeDef TIM_TimeBaseStructure;
  TIM_TimeBaseStructure.TIM_Period = 1000 - 1;       // 计数 1000 次即为 1ms
  TIM_TimeBaseStructure.TIM_Prescaler = 72 - 1;      // 预分频 72
  TIM_TimeBaseStructure.TIM_ClockDivision = TIM_CKD_DIV1;
  TIM_TimeBaseStructure.TIM_CounterMode = TIM_CounterMode_Up;
  TIM_TimeBaseInit(TIM1, &TIM_TimeBaseStructure);

  //开启更新中断
  TIM_ITConfig(TIM1, TIM_IT_Update, ENABLE);

  //配置中断优先级 (NVIC)
  NVIC_InitTypeDef NVIC_InitStructure;
  NVIC_InitStructure.NVIC_IRQChannel = TIM1_UP_IRQn;
  NVIC_InitStructure.NVIC_IRQChannelPreemptionPriority = 0; // 最高抢占优先级
  NVIC_InitStructure.NVIC_IRQChannelSubPriority = 1;
  NVIC_InitStructure.NVIC_IRQChannelCmd = ENABLE;
  NVIC_Init(&NVIC_InitStructure);

  TIM_Cmd(TIM1, ENABLE);
}

void TIM1_UP_IRQHandler()
{
    if (TIM_GetITStatus(TIM1, TIM_IT_Update) != RESET)
			{
        Timer_Tick++;//1ms自增一次  
        TIM_ClearITPendingBit(TIM1, TIM_IT_Update);
			}
}
