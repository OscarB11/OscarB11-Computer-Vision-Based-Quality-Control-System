   M S B A M L   `   `   `  ���� a  ]PresentationFramework.Aero, Version=4.0.0.0, Culture=neutral, PublicKeyToken=31bf3856ad364e35gJclr-namespace:Microsoft.Windows.Themes;assembly=PresentationFramework.AeroMicrosoft.Windows.Themes  R NWindowsBase, Version=4.0.0.0, Culture=neutral, PublicKeyToken=31bf3856ad364e35W SPresentationCore, Version=4.0.0.0, Culture=neutral, PublicKeyToken=31bf3856ad364e35\ XPresentationFramework, Version=4.0.0.0, Culture=neutral, PublicKeyToken=31bf3856ad364e35R NSystem.Xaml, Version=4.0.0.0, Culture=neutral, PublicKeyToken=b77a5c561934e089   DGHeaderColor  �� Color  DGColumnHeaderBorderBrush $  DGHeaderHorizontalGridLinesBrush  DGSelectedRow  TransparantColor  DGCellForeground  DGHeaderForeground  ColumHeaderStyle �� 
TargetType=   7System.Windows.Controls.Primitives.DataGridColumnHeader �� Property �� Value �� 
TargetType3   -Microsoft.Windows.Themes.DataGridHeaderBorder �� CornerRadius    CanUserSort   IsClickable   	IsPressed	 �� Padding
    SortDirection   SortDirection    SeparatorVisibility   SeparatorVisibility �� Triggers �� Binding �� ElementName �� Value �� Orientation J� SnapsToDevicePixels  DGTaskCellStyle*  $System.Windows.Controls.DataGridCell ��Name �� Triggers P� Property P� Value 	 DataGridRowStyle)  #System.Windows.Controls.DataGridRow?  9System.Windows.Controls.Primitives.DataGridCellsPresenter   
ItemsPanel �� 
TargetName 
 MainDataGridStyle&   System.Windows.Controls.DataGrid ��Name � ColumnDefinitions � RowDefinitions   SelectAllCommand �� RelativeSource �� AncestorType  �� 
ResourceId! �� TypeInTargetAssembly7  1System.Windows.Controls.DataGridHeadersVisibility
"   All# �� ConverterParameter!$   HeadersVisibilityConverter% �� 	ConverterG  ASystem.Windows.Controls.Primitives.DataGridColumnHeadersPresenter&  Name'   Column( ��Name) �� CanContentScroll* ��Name+ �� ScrollableHeight, � Maximum- �� Orientation. �� ViewportHeight/ �� ViewportSize*0 �� #ComputedVerticalScrollBarVisibility1 � Value2 �� Mode3 �� ScrollableWidth4 �� ViewportWidth,5 �� %ComputedHorizontalScrollBarVisibility  MainThemeRegionColor6 �� BasedOn  ProxyDColumHeaderStyle  ProxyDGStyle  StandardBorderBrush  StandardBackgroundBrush  HoverBorderBrush  SelectedBackgroundBrush  SelectedForegroundBrush  DisabledForegroundBrush  NormalBrush  NormalBorderBrush  HorizontalNormalBrush  HorizontalNormalBorderBrush  ListBoxBackgroundBrush7 �� 
StartPoint8 �� EndPoint9 � Color: � Offset  StandardBrush  
GlyphBrush  PressedBrush  ScrollBarLineButton # MainThemeControlBackgroundColor  MainThemeControlBorderBrush; �� Path  ScrollBarPageButton   ScrollBarThumb ! VerticalScrollBar< ]� DecreaseRepeatButton= ]� Thumb> ]� IncreaseRepeatButton " HorizontalScrollBar # ScollViewerTemplate? �Name@ ��NameA �� CanHorizontallyScrollB �� CanVerticallyScroll4  .System.Windows.Automation.AutomationPropertiesC   AutomationIdD � CursorE � Minimum�� D 9http://schemas.microsoft.com/winfx/2006/xaml/presentation    :x,http://schemas.microsoft.com/winfx/2006/xaml     UThemesJclr-namespace:Microsoft.Windows.Themes;assembly=PresentationFramework.Aero  %�"  &	        &	      &	 &     &	 7     &	 J     &	 a     &	 t     &	 �     7  &	 �    &		 ~    &	
 �    7   7  '  y    7 
 &	 �    7  &	 �    7 
 7   7  &	 R    &	 b    &	 r    &	 �    &	 �    &	 �    &	 �    &	 �    &	 �    &	 �    &	     &	 �    &	 �    &	 �    &	 |    7  7  7  7  &	     &	  �    &	! ~    7  7  7   7  7  7  &	" E    7  7  7   7  7  7  '��     7 " 7 ! &	# >    �� $  #2D354D���� $  #B2B2B2���� $  Black���� $  #D5E5F7���� $  Transparent���� $  #262626���� $  #707070���� "   .���  � ��# �   ��  � ��$ Center=���  � ��$
 Left=���  � ��$ Stretch=���  � ��$ 48����  � ��8   ��  � ��$ 14(���  � ��$
 Bold"���  � ��$
 45,0q���  � �� �� "   .R�� .+�  #���  $��0,0,0,0q�$ 	10,10,0,0��$��0,0,0,0q�# z # z��#	 z��# z
 # z ���� "  .���  � ��# �    ��  �� *
Visibility+$ 	SortArrow�� Visible._�� .+��� $ 
Horizontal=�$��Center=�.+��� $��0��# z $��0,0,0,0q��� "  .���  � ��$
 Left=���  � �� �� "  .R��� $ border��
������ #���  $��1q� . ._��� # z $
��Left=�$��Center=�$	��5,0q� P�  �@ 
IsSelected . . ���  � ��
 ������� "  .���  �@J�SnapsToDevicePixels . ��  � ��$
 Left=���  � ��$ 60����  �@=�ErrorTemplate Z� ��  � �� �� "  .R��� $ 
DGR_Border�����#1c1e27$ 10�� . $��0,0,0,12q�._�� $��0,0,0,0q�.+�  $��1��#��z # z  P�  �@J�IsMouseOver . . ��� $ 
DGR_Border�� � �� �#3A5485�� "  .���  � ��# �  ��  �@ CanUserReorderColumns .  ��  � ��$ Stretch=���  � ��
 �������  � ��$ 14(���  � ��$ Regular"���   �@ RowDetailsVisibilityMode$ VisibleWhenSelected=���  � �� . ��  �@��PanningMode$
 Both=���  �@ AutoGenerateColumns .  ��  �@ GridLinesVisibility$
 None=���   �@ HorizontalGridLinesBrush
 � � ���  �@ VerticalGridLinesBrush
 ���� ��  �@ 
IsReadOnly . ��  �@ 	RowHeight$ 28Q���  �@ CanUserAddRows .  ��  �@ RowHeaderWidth$ 0Q���  � �� . ��  � ��$
 Auto=���  � ��$
 Auto=���  �@ ColumnHeaderStyle# �  ��  �@ RowStyle# � 	 ��  �@ 	CellStyle# �  ��  �@ AutoGenerateColumns .  ��  �@ HeadersVisibility$ Column=���  �@ CanUserAddRows .  ��  �@ EnableRowVirtualization .  �� " �@ EnableColumnVirtualization .  ��  � �� �� "  .R��� #���  $��0q�$ 0��#	 z�� . ._��� $ DG_ScrollViewer��|�.  ���� " ��.R��  �� $
��Auto ��� $��* ��� $
��Auto � �� $
��Auto ��� $��* ��� $
��Auto �.+��� #��Z |�.  ���� *CellsPanelHorizontalOffset+ �� "  ��C� *�� $"  DataGridSelectAllButtonStyle��"!  +y��� *HeadersVisibility+## Z" #% Z$  �� "    $!& PART_ColumnHeadersPresenter��$��1��$��2��y��� *HeadersVisibility+## Z' #% Z$  �� "  �� $!( PART_ScrollContentPresenter��$��1��$��2��#) z���� $* PART_VerticalScrollBar��$��1��$��2��$��Right=�#, z+ $��0,5,8,0q�$- Vertical=�#/ z. #y�z0 1 �� *VerticalOffset+$2 OneWay=� �� *TemplatedParent+� $��2��$��1�� �� ���� **(NonFrozenColumnsViewportHorizontalOffset+ �� "  �� $��* �.+��� $* PART_HorizontalScrollBar��$��1��#, z3 $- 
Horizontal=�#/ z4 #y�z5 $��0,0,8,8q�1 �� *HorizontalOffset+$2 OneWay=� �� *TemplatedParent+.��� 8��  .+��� 8�� $ 	20,20,0,0��$
	 45,0q�._��� # z $��	0,15,0,15q� P�  �@��
IsGrouping . . ���  � �� . �� 86   "  �� "   .���  � ��# �   ��  � ��$ Center=���  � ��$
 Left=���  � ��$ Stretch=���  � ��$ 48����  � ��8   ��  � ��$ 14(���  � ��$
 Bold"���  � ��$
 30,0q���  � �� �� "   .R�� .+�  #���  $��0,0,0,0q�$ 	10,10,0,0��$��0,0,0,0q�# z # z��#	 z��# z
 # z ���� "  .���  � ��# �    ��  �� *
Visibility+$ 	SortArrow�� Visible._�� .+��� $ 
Horizontal=�$��Center=�.+��� $��0��# z $��0,0,0,0q��� "  86   .���  �@ ColumnHeaderStyle# �  ��  � �� �� "  .R��� #���  $��0q�$ 0��#	 z�� . ._��� $ DG_ScrollViewer��|�.  ���� " ��.R��  �� $
��Auto ��� $��* ��� $
��Auto � �� $
��Auto ��� $��* ��� $
��Auto �.+��� #��Z |�.  ���� *CellsPanelHorizontalOffset+ �� "  ��C� *�� $"  DataGridSelectAllButtonStyle��"!  +y��� *HeadersVisibility+## Z" #% Z$  �� "    $!& PART_ColumnHeadersPresenter��$��1��$��2��y��� *HeadersVisibility+## Z' #% Z$  �� "  �� $!( PART_ScrollContentPresenter��$��1��$��2��#) z���� $* PART_VerticalScrollBar��$��1��$��2��$��Right=�#, z+ $��0,5,8,0q�$- Vertical=�#/ z. #y�z0 1 �� *VerticalOffset+$2 OneWay=� �� *TemplatedParent+� $��2��$��1��$��2�� �� ���� **(NonFrozenColumnsViewportHorizontalOffset+ �� "  �� $��* �.+��� $* PART_HorizontalScrollBar��$��1��#, z3 $��0,0,0,8q�$- 
Horizontal=�#/ z4 #y�z5 1 �� *HorizontalOffset+$2 OneWay=� �� *TemplatedParent+.��� 8�� .+��� 8�� $ 	20,20,0,0��$
	 30,0q�._��� # z $��	0,15,0,15q��� $
  #888���� $
  #FFF���� $
  #DDD���� $
  Gray���� $  White���� $
  #888���� $  Transparent���� $  DarkSlateBlue���� $  Transparent���� $  DarkSlateBlue���� $	7 0,0&�$8 1,0.001&���� � $9 White��$	: 0.0Q�� $9 White��$	: 0.6Q�� $9 #DDDDDD��$	: 1.2Q��� $	7 0,0&�$	8 0,1&���� � $
9 #FFF��$	: 0.0Q�� $
9 #CCC��$	: 1.0Q��� $
  #444���� $	7 0,0&�$	8 0,1&���� � $
9 #BBB��$	: 0.0Q�� $
9 #EEE��$	: 0.1Q�� $
9 #EEE��$	: 0.9Q�� $
9 #FFF��$	: 1.0Q��� " ��.���  �@J�SnapsToDevicePixels . ��  �@�OverridesDefaultStyle . ��  � |� .  ��  � �� �� " ��.R��� $��Border��$��1q�$ 8��8��  #���  $��1q�._�F� $��Center=�$��Center=�8�� ���� $; Content� �� *TemplatedParent+ P�  � �� . . ��� $ Border��8   � ��P�  � {� .  . ��� 8   � ���� " ��.���  �@J�SnapsToDevicePixels . ��  �@�OverridesDefaultStyle . ��  � �� .  ��  � |� .  ��  � �� �� " ��.R��� 
������ �� " n�.���  �@J�SnapsToDevicePixels . ��  �@�OverridesDefaultStyle . ��  � �� .  ��  � |� .  ��  � �� �� " n�.R��� $ 4��#��z��#��z��$��2q��� " ��.R��  �� $��14���� $��0.00001* ��� $��14��.+��� $��3��$ 4��#���  $��0q�
������ �� $��0��8��  $��14��$��ScrollBar.LineUpCommand��$��M 0 4 L 8 4 L 4 0 Z��]� $��
PART_Track��$��1����. < �� 8�� $��ScrollBar.PageUpCommand��= n� 8�� $��1,0,1,0q�8�� #���  > �� 8�� $��ScrollBar.PageDownCommand���� $��3��8�� $��14��$��ScrollBar.LineDownCommand��$��M 0 0 L 4 4 L 8 0 Z���� " ��.R��  �� $��14���� $��0.00001* ��� $��14��.+��� $��3��$ 2��
������ �� $��0��8��  $��14��$��ScrollBar.LineLeftCommand��$��M 4 0 L 4 8 L 0 4 Z��]� $��
PART_Track��$��1����.  < �� 8�� $��ScrollBar.PageLeftCommand��= n� 8�� $��0,1,0,1q�8�� #���  > �� 8�� $ ��ScrollBar.PageRightCommand���� $��3��8�� $��14��$ ��ScrollBar.LineRightCommand��$��M 0 0 L 4 4 L 0 8 Z���� " ��.���  �@J�SnapsToDevicePixels . ��  �@�OverridesDefaultStyle .  P�  �@��Orientation$ 
Horizontal=�. ���  � ��$
 Auto����  � ��$ 14����  � ��8   P�  �@��Orientation$ Vertical=�. ���  � ��$ 14����  � ��$
 Auto����  � ��8  �� " ��.R�� $
? Grid��#��z�� �� $��* ��� $
��Auto � �� $��* ��� $
��Auto �.+��� $@ Corner��$��1��#��� ��$��1���� $!( PART_ScrollContentPresenter��#) z��A .  B .  #��z��#��z��$��0��#��z��$��0���� $* PART_VerticalScrollBar��$C VerticalScrollBar��$D Arrow��$��1��#, z+ $E 0Q�$��0��#y�z0 #/ z. $��10,20,10,20q�1 �� *VerticalOffset+$2 OneWay=� �� *TemplatedParent+�� $* PART_HorizontalScrollBar��$C HorizontalScrollBar��$D Arrow��$��0��#, z3 $E 0Q�$- 
Horizontal=�$��1��#y�z5 #/ z4 1 �� *HorizontalOffset+$2 OneWay=� �� *TemplatedParent+