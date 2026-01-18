# 2026 CSDCMS CanSat Competition: Coudln'tSat

This repository will contain the entirety of the Couldn'tSat team's submission to the [2026 CSDCMS CanSat Competition](https://www.csdcms.ca/CanSat_info.html).

As specified in the [primary mission](https://www.csdcms.ca/Docs/CSDC_CanSat_Requirements_2024-25.pdf#page=7), our CanSat will collect air temperature and presssure data at a rate of no less than 1 Hz, and transmit this data to a ground station using LoRa radio.

Our CanSat has two main parts to its [secondary mission](https://www.csdcms.ca/Docs/CSDC_CanSat_Requirements_2024-25.pdf#page=7):
- Using a custom trained Monocular Height Estimation (MHE) model, it will use a single photo from an onboard downwards facing camera to generate a Digital Surface Map (DSM) of the launch site terrain.
- The CanSat will then use the MHE-generated DSM to determine a flat landing spot, and perform a guided landing using a paraglider.

At the moment, this repository contains the beginnings of the code for our MHE model.