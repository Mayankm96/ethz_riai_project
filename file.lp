\ Model LP
\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  0 x[0] + 0 x[1] + 0 x[3] + 0 x[4] + 0 x[5] + 0 x[6] + 0 x[7] + 0 x[8]
   + 0 x[9] + 0 x[10] + 0 x[12] + 0 x[14] + 0 x[15] + 0 x[16] + 0 x[17]
   + 0 x[18] + 0 x[19] - 0.5167534947395325 relu_out[0]
   - 0.0393579155206681 relu_out[1] + 0.1267701387405396 relu_out[2]
   + 0.3251743316650391 relu_out[3] - 0.0476175397634507 relu_out[4]
   - 0.557050108909607 relu_out[5] - 0.0661850571632386 relu_out[6]
   - 0.2286860346794129 relu_out[7] + 0.7014030218124391 relu_out[8]
   - 0.5232732295989991 relu_out[9] - 0.0682687237858773 relu_out[10]
   - 0.8181120157241822 relu_out[11] + 0.7986193895339966 relu_out[12]
   - 0.0205628443509341 relu_out[13] - 0.4577025473117829 relu_out[14]
   - 0.2937511205673218 relu_out[15] - 0.1102422773838044 relu_out[16]
   - 0.4495534896850586 relu_out[17] + 0.7265208959579468 relu_out[18]
   - 0.1826855093240738 relu_out[19] - 0.3051835894584656 Constant
Subject To
 nontrivial_zero_lb[2]: relu_out[2] >= 0
 nontrivial_zero_lb[11]: relu_out[11] >= 0
 nontrivial_zero_lb[13]: relu_out[13] >= 0
 nontrivial_variable_lb[2]: - x[2] + relu_out[2] >= 0
 nontrivial_variable_lb[11]: - x[11] + relu_out[11] >= 0
 nontrivial_variable_lb[13]: - x[13] + relu_out[13] >= 0
 nontrivial_line_ub[2]: - 0.9954610236849228 x[2] + relu_out[2]
   <= 0.00608367208160736
 nontrivial_line_ub[11]: - 0.6356630220698036 x[11] + relu_out[11]
   <= 0.3529814754432344
 nontrivial_line_ub[13]: - 0.5126510936501716 x[13] + relu_out[13]
   <= 0.3809560367929523
 trivial_lb[0]: relu_out[0] >= 0
 trivial_lb[1]: relu_out[1] >= 0
 trivial_lb[3]: relu_out[3] >= 1.193807624270096
 trivial_lb[4]: relu_out[4] >= 0
 trivial_lb[5]: relu_out[5] >= 0.0250781994784521
 trivial_lb[6]: relu_out[6] >= 0
 trivial_lb[7]: relu_out[7] >= 0
 trivial_lb[8]: relu_out[8] >= 0
 trivial_lb[9]: relu_out[9] >= 2.118299335483569
 trivial_lb[10]: relu_out[10] >= 2.854224866650947
 trivial_lb[12]: relu_out[12] >= 3.182348378157549
 trivial_lb[14]: relu_out[14] >= 6.405014021943324
 trivial_lb[15]: relu_out[15] >= 0
 trivial_lb[16]: relu_out[16] >= 0
 trivial_lb[17]: relu_out[17] >= 0
 trivial_lb[18]: relu_out[18] >= 1.278311268785439
 trivial_lb[19]: relu_out[19] >= 0
 trivial_ub[0]: relu_out[0] <= 0
 trivial_ub[1]: relu_out[1] <= 0
 trivial_ub[3]: relu_out[3] <= 2.797381800848076
 trivial_ub[4]: relu_out[4] <= 0
 trivial_ub[5]: relu_out[5] <= 1.108107429743827
 trivial_ub[6]: relu_out[6] <= 0
 trivial_ub[7]: relu_out[7] <= 0
 trivial_ub[8]: relu_out[8] <= 0
 trivial_ub[9]: relu_out[9] <= 3.506366330997561
 trivial_ub[10]: relu_out[10] <= 3.347992039815794
 trivial_ub[12]: relu_out[12] <= 4.007850341181373
 trivial_ub[14]: relu_out[14] <= 7.665733029324332
 trivial_ub[15]: relu_out[15] <= 0
 trivial_ub[16]: relu_out[16] <= 0
 trivial_ub[17]: relu_out[17] <= 0
 trivial_ub[18]: relu_out[18] <= 2.957955248762785
 trivial_ub[19]: relu_out[19] <= 0
Bounds
 -11.37422431573999 <= x[0] <= -10.35110221201105
 -1.228116399887789 <= x[1] <= -0.7020420031531673
 -0.00611141163426698 <= x[2] <= 1.340318093619261
 1.193807624270096 <= x[3] <= 2.797381800848076
 -4.489588388070387 <= x[4] <= -3.456216085220677
 0.0250781994784521 <= x[5] <= 1.108107429743827
 -2.653201931205253 <= x[6] <= -1.706173003983357
 -4.806547845594529 <= x[7] <= -3.613736184854918
 -3.920619893039226 <= x[8] <= -2.915856712177697
 2.118299335483569 <= x[9] <= 3.506366330997561
 2.854224866650947 <= x[10] <= 3.347992039815794
 -0.5552965379264623 <= x[11] <= 0.9688324184070665
 3.182348378157549 <= x[12] <= 4.007850341181373
 -0.7431097709759556 <= x[13] <= 0.7816905544043524
 6.405014021943324 <= x[14] <= 7.665733029324332
 -11.2705680973902 <= x[15] <= -9.668450723363696
 -8.815962388534995 <= x[16] <= -7.397721649090278
 -4.455024471978499 <= x[17] <= -3.006805962285971
 1.278311268785439 <= x[18] <= 2.957955248762785
 -4.303819481761722 <= x[19] <= -3.374931720618608
 Constant = 1
End
