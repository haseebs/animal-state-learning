# animal_state

1. Reinstall using requirements.txt
2. Run `wandb login` on shell and paste your API key (or paste the one provided
   here if you dont have an account): `81265a4b037275d39939b0dc2787f6673a25223e`
3. Default params are given in `config-defaults.yaml`. Sweep params are provided in `sweep.yaml`. If you change anything in `sweep.yaml`, run `wandb sweep --update hshah1/animal-state/xst380tx sweep.yaml`.
4. Run `wandb agent hshah1/animal-state/xst380tx`. Each instance of this
   command will run one experiment at a time. So for 10 parallel agents, start 10 instances of this command.

Server errors are fine, just wait a minute for them to get fixed.
