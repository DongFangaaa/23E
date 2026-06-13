#include "stm32f10x.h"                  // Device header
#include "PWM.h"

void Motor_Init()
{
	RCC_APB2PeriphClockCmd(RCC_APB2Periph_GPIOA, ENABLE);
	
	GPIO_InitTypeDef GPIO_InitStructure;
	GPIO_InitStructure.GPIO_Mode = GPIO_Mode_Out_PP;
	GPIO_InitStructure.GPIO_Pin = GPIO_Pin_2 | GPIO_Pin_3 | GPIO_Pin_4 | GPIO_Pin_5;
	GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
	GPIO_Init(GPIOA, &GPIO_InitStructure);
	
	GPIO_SetBits(GPIOA,GPIO_Pin_3 | GPIO_Pin_5);
	GPIO_ResetBits(GPIOA,GPIO_Pin_2 | GPIO_Pin_4);
	
	PWM_Init();
}

void Mptor_SetSpeed(int16_t left_speed,int16_t right_speed)
{
	if (left_speed >= 0)
		{
      GPIO_SetBits(GPIOA, GPIO_Pin_3); 
			GPIO_ResetBits(GPIOA, GPIO_Pin_2); // 正转
      PWM_SetCompare_1(left_speed);
    } 
		else 
		{
      GPIO_SetBits(GPIOA, GPIO_Pin_2);
			GPIO_ResetBits(GPIOA, GPIO_Pin_3); // 反转
      PWM_SetCompare_1(-left_speed); // 占空比必须是正数
    }
		if (right_speed >= 0)
		{
      GPIO_SetBits(GPIOA, GPIO_Pin_5); 
			GPIO_ResetBits(GPIOA, GPIO_Pin_4); // 正转
      PWM_SetCompare_2(right_speed);
    } 
		else 
		{
      GPIO_SetBits(GPIOA, GPIO_Pin_4);
			GPIO_ResetBits(GPIOA, GPIO_Pin_5); // 反转
      PWM_SetCompare_2(-right_speed); // 占空比必须是正数
    }
}
