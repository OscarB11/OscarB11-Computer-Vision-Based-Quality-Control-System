   M S B A M L   `   `   `  ����E  APCB_Vision, Version=1.0.0.0, Culture=neutral, PublicKeyToken=null3clr-namespace:PCB_Vision.ViewsPCB_Vision.Views  5   	   "    PCB_Vision.Views.InputWindow   5   	   R NWindowsBase, Version=4.0.0.0, Culture=neutral, PublicKeyToken=31bf3856ad364e35W SPresentationCore, Version=4.0.0.0, Culture=neutral, PublicKeyToken=31bf3856ad364e35\ XPresentationFramework, Version=4.0.0.0, Culture=neutral, PublicKeyToken=31bf3856ad364e35D 9http://schemas.microsoft.com/winfx/2006/xaml/presentation    5   	   R NSystem.Xaml, Version=4.0.0.0, Culture=neutral, PublicKeyToken=b77a5c561934e089:x,http://schemas.microsoft.com/winfx/2006/xaml     5   	   @d2http://schemas.microsoft.com/expression/blend/2008     5   	   Jmc;http://schemas.openxmlformats.org/markup-compatibility/2006     5   	   0localclr-namespace:PCB_Vision.Views     5   	   
������   � Title$   ��5   	    � WindowStartupLocation$ CenterScreen=�6    � 
ResizeMode$ NoResize=�67   $	��300��5	   	   $	��250��6    � Topmost .  � 	Resources 5
       �� 
TargetType   
LabelStyle �� Property �� Value  MainForeground��%�   &	        7  �� 5   
   " ��6   .�5      ��  � ��$ 14(�6)   �� 5       � ��$
 Bold"�6+   �� 5       � ��8   6+   �� 5       � ��$ Center=�62   5      5      .��5      �   Screen.Background#��[ 6   .+�5   
   � $��Stretch=�6   $��Center=�6-   $��15,0,0,0q�6H    � ColumnDefinitions 5      �� 5      $��* �6#   �� 5      $��23 �6#   	 � RowDefinitions	 5      �� 5      $
��auto �6    �� 5      $
��auto �6    �� 5      $
��auto �6    .+�5      �� $��Enter Input��6   #��[  6+   �� 5      

 � Tag$
 Type Something...��6     TextBoxStyle#��[ 6.     MainThemeItemColor#��[ 5        MainThemeItemBorderColor#��[ 5       $��1��6;   $��0,12,0,0q�6H   $��51��6Z   ��5   4   �� *OperatorUserObj.FullName+ �� Mode$ TwoWay=� �� UpdateSourceTrigger$ PropertyChanged=�5    Z   �� 5!      -   5         ��Name$ btnOkay��5!      $��3��6'   $��Ok��64   $��50��6A   $��Right=�6M   $��80��5"      $
��0,10q�6"     TopButtonStyle#��[ 60   ��5%      �� 5&       �� ImageSource$4 ./PCB_Vision;component/Resources/Images/Que.png��6!    j� Stretch$ UniformToFill=�6^   5(      5)      5*      5+      