#include "stm32f10x.h"                  // Device header

void PWM_Init()
{
	RCC_APB1PeriphClockCmd(RCC_APB1Periph_TIM2,ENABLE);
	RCC_APB2PeriphClockCmd(RCC_APB2Periph_GPIOA,ENABLE);
	
	GPIO_InitTypeDef GPIO_Initstructure;
	GPIO_Initstructure.GPIO_Mode = GPIO_Mode_AF_PP;
	GPIO_Initstructure.GPIO_Pin = GPIO_Pin_0 |GPIO_Pin_1;
	GPIO_Initstructure.GPIO_Speed = GPIO_Speed_50MHz;
	GPIO_Init(GPIOA,&GPIO_Initstructure);
	
	TIM_InternalClockConfig(TIM2);
	
	TIM_TimeBaseInitTypeDef TIM_TimeBaseInitstructure;
	TIM_TimeBaseInitstructure.TIM_ClockDivision = TIM_CKD_DIV1;
	TIM_TimeBaseInitstructure.TIM_CounterMode = TIM_CounterMode_Up;
	TIM_TimeBaseInitstructure.TIM_Period = 100 - 1;
	TIM_TimeBaseInitstructure.TIM_Prescaler = 36 - 1;
	TIM_TimeBaseInitstructure.TIM_RepetitionCounter = 0;
	TIM_TimeBaseInit(TIM2,&TIM_TimeBaseInitstructure);//20ms
	
	TIM_OCInitTypeDef TIM_OCInitstructure;
	TIM_OCStructInit(&TIM_OCInitstructure);
	TIM_OCInitstructure.TIM_OCMode = TIM_OCMode_PWM1;
	TIM_OCInitstructure.TIM_OCPolarity = TIM_OCPolarity_High;
	TIM_OCInitstructure.TIM_OutputState = TIM_OutputState_Enable;
	TIM_OCInitstructure.TIM_Pulse = 0;
	TIM_OC1Init(TIM2,&TIM_OCInitstructure);
	TIM_OC2Init(TIM2,&TIM_OCInitstructure);
	
	TIM_Cmd(TIM2,ENABLE);
}

void PWM_SetCompare_1(uint16_t Compare1)
{
	TIM_SetCompare1(TIM2,Compare1);
}

void PWM_SetCompare_2(uint16_t Compare2)
{
	TIM_SetCompare2(TIM2,Compare2);
}
