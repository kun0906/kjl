"""
pcap statistics
"""
import matplotlib.pyplot as plt
from odet.pparser.parser import PCAP, _get_fid, _get_frame_time
from scapy.layers.inet import TCP, UDP
from scapy.utils import PcapReader


class Pcap(PCAP):

    def __init__(self, in_file=''):
        self.in_file = in_file

    def pcap2flows(self):
        pp = PCAP(pcap_file=self.in_file)
        pp.pcap2flows()
        self.flows = pp.flows

    # def src2dsts(self):
    #
    #     src2dsts_mappings = {}
    #     for (fid, pkts) in self.flows:
    #         src = fid[0]
    #         if src not in src2dsts_mappings.keys():
    #             src2dsts_mappings[src] = pkts
    #         else:
    #             src2dsts_mappings[src].append(pkts)
    #     return src2dsts_mappings
    #
    # def dst2srcs(self):
    #
    #     dst2srcs_mappings = {}
    #     for (fid, pkts) in self.flows:
    #         dst = fid[0]
    #         if dst not in dst2srcs_mappings.keys():
    #             dst2srcs_mappings[dst] = pkts
    #         else:
    #             dst2srcs_mappings[dst].append(pkts)
    #
    #     return dst2srcs_mappings

    def pcap2pkts_stats(self):
        src_pkts = {}
        dst_pkts = {}
        verbose = 10
        try:
            # iteratively get each packet from the pcap
            for i, pkt in enumerate(PcapReader(self.in_file)):
                if (verbose > 3) and (i % 10000 == 0):
                    print(f'ith_packets: {i}')

                if (TCP in pkt) or (UDP in pkt):
                    # this function treats bidirection flows as two sessions (hereafter, we use sessions
                    # and flows interchangeably).
                    fid = _get_fid(pkt)
                    src = fid[0]  # only src
                    dst = fid[1]
                    if src not in src_pkts.keys():
                        src_pkts[src] = [pkt]
                    else:
                        src_pkts[src].append(pkt)

                    if dst not in dst_pkts.keys():
                        dst_pkts[dst] = [pkt]
                    else:
                        dst_pkts[dst].append(pkt)

                else:
                    continue

        except Exception as e:
            msg = f'Parse PCAP error: {e}!'
            raise RuntimeError(msg)

        self.src_pkts = src_pkts
        self.dst_pkts = dst_pkts

        return self

    def _show(self, ax, x, y, label, xlabel, ylabel='', title='', is_xlabel=True, is_ylabel=True):

        ax.plot(x, y, '*-', alpha=0.9, label=label)
        # ax.plot(x, y, '*-', alpha=0.9)
        # ax.plot([0, 1], [0, 1], 'k--', label='', alpha=0.9)

        # # plt.xlim([0.0, 1.0])
        # if len(ylim) == 2:
        #     plt.ylim(ylim)  # [0.0, 1.05]
        if is_xlabel:
            ax.set_xlabel(xlabel)
        else:
            ax.get_xaxis().set_visible(False)
        if is_ylabel:
            ax.set_ylabel(ylabel)
        else:
            # ax.get_yaxis().set_visible(False)
            ax.set_ylabel('')
        # # plt.xticks(x)
        # # plt.yticks(y)
        ax.legend(loc='lower right', fontsize=5)
        # plt.title(title)

    def show(self, src_ptks, name = 'src', out_file='res.pdf'):

        # only show top_5
        top_num = 5
        nums =sorted([len(pkts) for key, pkts in src_ptks.items()], reverse=True)
        thres = nums[top_num-1]
        tops = [(key, pkts) for key, pkts in src_ptks.items() if len(pkts) >=thres]
        print([(name, key, len(pkts)) for key, pkts in tops])

        def split_pkts(pkts, start=0, interval=1):
            split_pkts_info = []
            # start = _get_frame_time(pkts[0])
            i = 0
            cnt = 0
            while i < len(pkts):
                cur = _get_frame_time(pkts[i])
                if cur - start <= interval:
                    cnt += 1
                else:
                    split_pkts_info.append((start, cnt))
                    cnt = 1
                    start = cur
                i += 1

            return split_pkts_info

        n_rows = top_num
        fig, ax = plt.subplots(nrows=n_rows, ncols=1)
        ax = ax.reshape(-1, 1)

        for i, (ip, pkts) in enumerate(tops):
            if i == 0:
                start = _get_frame_time(pkts[0])
            split_pkts_info = split_pkts(pkts, start=start, interval=1)
            x = [start_time for start_time, n_pkts in split_pkts_info]
            y = [n_pkts for start_time, n_pkts in split_pkts_info]
            ylabel = f'Num. of pkts'
            is_xlabel = False
            if i == len(tops) - 1:
                is_xlabel = True
            self._show(ax[i, 0], x, y, xlabel='', ylabel=ylabel, label=f'{name}:{ip}', title='', is_xlabel=is_xlabel, is_ylabel=False)

        # fig.text(0.5, 0.04, xlabel, ha='center')
        # fig.text(0.04, 0.5, ylabel, va='center', rotation='vertical')

        xlabel = 'Time (interval=1s)'
        title = f'{name}'
        fig.suptitle(title, fontsize=11)

        plt.tight_layout()  # rect=[0, 0, 1, 0.95]
        try:
            plt.subplots_adjust(top=0.92, bottom=0.1, right=0.975, left=0.12)
        except Warning as e:
            raise ValueError(e)
        #
        # fig.text(.5, 15, "total label", ha='center')
        plt.figtext(0.5, 0.01, f'{xlabel}', fontsize=11, va="bottom", ha="center")
        print(out_file)
        fig.savefig(out_file, format='pdf', dpi=300)
        plt.show()
        plt.close(fig)

        ###################################################################################
        # all in one figure
        fig, ax = plt.subplots(nrows=1, ncols=1)

        for i, (ip, pkts) in enumerate(tops):
            if i == 0:
                start = _get_frame_time(pkts[0])
            split_pkts_info = split_pkts(pkts, start=start, interval=1)
            x = [start_time for start_time, n_pkts in split_pkts_info]
            y = [n_pkts for start_time, n_pkts in split_pkts_info]
            ylabel = f'Num. of pkts'
            self._show(ax, x, y, xlabel='', ylabel=ylabel, label=f'{name}:{ip}', title='', is_xlabel=True, is_ylabel=True)

        xlabel = 'Time (interval=1s)'
        title = f'{name}'
        fig.suptitle(title, fontsize=11)

        plt.tight_layout()  # rect=[0, 0, 1, 0.95]
        try:
            plt.subplots_adjust(top=0.92, bottom=0.1, right=0.975, left=0.12)
        except Warning as e:
            raise ValueError(e)
        #
        # fig.text(.5, 15, "total label", ha='center')
        plt.figtext(0.5, 0.01, f'{xlabel}', fontsize=11, va="bottom", ha="center")
        print(out_file)
        fig.savefig(out_file, format='pdf', dpi=300)
        plt.show()
        plt.close(fig)

def main():
    in_file = 'speedup/data/deeplens_open_shut_fridge_batch_8.pcap_filtered.pcap'
    pp = Pcap(in_file=in_file)
    pp.pcap2pkts_stats()

    pp.show(pp.src_pkts, name='src', out_file=pp.in_file + '-src.pdf')
    pp.show(pp.dst_pkts, name ='dst', out_file=pp.in_file + '-dst.pdf')


if __name__ == '__main__':
    main()
