#include "stm32f10x.h"                  // Device header
#include "HCSR04.h"

extern volatile uint32_t Timer_Tick; 

static uint32_t Buzzer_EndTime = 0; 

void Buzzer_Init()
{
	RCC_APB2PeriphClockCmd(RCC_APB2Periph_GPIOA,ENABLE);
	
	GPIO_InitTypeDef GPIO_Initstructure;
	GPIO_Initstructure.GPIO_Mode = GPIO_Mode_Out_PP;
	GPIO_Initstructure.GPIO_Pin = GPIO_Pin_6;
	GPIO_Initstructure.GPIO_Speed = GPIO_Speed_50MHz;
	GPIO_Init(GPIOA,&GPIO_Initstructure);
	
	GPIO_SetBits(GPIOA, GPIO_Pin_6);//关
}

void Buzzer_Beep()
{
	if(Buzzer_EndTime != 0)
	{
		return;
	}
    Buzzer_EndTime = Timer_Tick + 600;
    GPIO_ResetBits(GPIOA, GPIO_Pin_6); // 响
}

void Buzzer_Tick_Task()
{
    if (Buzzer_EndTime != 0 && Timer_Tick >= Buzzer_EndTime)
    {
        GPIO_SetBits(GPIOA, GPIO_Pin_6); // 停
        Buzzer_EndTime = 0;
    }
}
